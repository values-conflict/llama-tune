package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// ggufModel holds the model metadata the harness cares about.
type ggufModel struct {
	Path       string
	Version    int
	Arch       string
	BlockCount int64
	NEmbd      int64
	NContext   int64
	NHeads     int64 // attention_head_count
	NKVHeads   int64 // attention_kv_head_count (0 = MHA)
	SlidingWin int64 // attention_sliding_window, -1 if absent
	NExperts   int64 // ffn_expert_count, 0 if dense
	NExpertSel int64 // ffn_expert_weights, experts active per token
	FileType   string
	NTensors   int64
	HasNextN   bool // contains any .nextn. tensor
	MTP        bool // has blk.<block_count-1>.nextn.eh_proj.weight, i.e. draft-mtp capable
}

// scanGGUF reads only the GGUF metadata section (kv pairs + tensor info),
// never the (multi-GB) tensor data that follows.
func scanGGUF(path string) (*ggufModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReaderSize(f, 1<<20)
	m := &ggufModel{Path: path}

	var u32 = func() (uint32, error) {
		var b [4]byte
		if _, err := io.ReadFull(r, b[:]); err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint32(b[:]), nil
	}
	var u16 = func() (uint16, error) {
		var b [2]byte
		if _, err := io.ReadFull(r, b[:]); err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint16(b[:]), nil
	}
	var u64 = func() (uint64, error) {
		var b [8]byte
		if _, err := io.ReadFull(r, b[:]); err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint64(b[:]), nil
	}
	var readString = func() (string, error) {
		n, err := u64()
		if err != nil {
			return "", err
		}
		buf := make([]byte, n)
		if _, err := io.ReadFull(r, buf); err != nil {
			return "", err
		}
		return strings.TrimSuffix(string(buf), "\x00"), nil
	}
	var readValue func(uint32) (any, error)
	readValue = func(typ uint32) (any, error) {
		switch typ {
		case 0:
			b, err := r.ReadByte()
			return int64(b), err
		case 1:
			b, err := r.ReadByte()
			return int64(int8(b)), err
		case 2:
			v, err := u16()
			return int64(v), err
		case 3:
			v, err := u16()
			return int64(int16(v)), err
		case 4:
			v, err := u32()
			return int64(v), err
		case 5:
			v, err := u32()
			return int64(int32(v)), err
		case 6:
			v, err := u32()
			return float64(math.Float32frombits(v)), err
		case 7:
			b, err := r.ReadByte()
			return b != 0, err
		case 8:
			return readString()
		case 9:
			elem, err := u32()
			if err != nil {
				return nil, err
			}
			count, err := u64()
			if err != nil {
				return nil, err
			}
			arr := make([]any, count)
			for i := range arr {
				arr[i], err = readValue(elem)
				if err != nil {
					return nil, err
				}
			}
			return arr, nil
		case 10:
			v, err := u64()
			return int64(v), err
		case 11:
			v, err := u64()
			return int64(int64(v)), err
		case 12:
			v, err := u64()
			return math.Float64frombits(v), err
		default:
			return nil, fmt.Errorf("unsupported gguf value type %d", typ)
		}
	}
	var asInt = func(v any) int64 {
		switch n := v.(type) {
		case int64:
			return n
		case float64:
			return int64(n)
		}
		return 0
	}

	magic, err := u32()
	if err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != 0x46554747 { // "GGUF"
		return nil, fmt.Errorf("not a gguf file: %s", path)
	}
	ver, err := u32()
	if err != nil {
		return nil, err
	}
	m.Version = int(ver)

	nTensors, err := u64()
	if err != nil {
		return nil, err
	}
	nKV, err := u64()
	if err != nil {
		return nil, err
	}

	for i := uint64(0); i < nKV; i++ {
		key, err := readString()
		if err != nil {
			return nil, fmt.Errorf("read kv key: %w", err)
		}
		typ, err := u32()
		if err != nil {
			return nil, err
		}
		val, err := readValue(typ)
		if err != nil {
			return nil, fmt.Errorf("read kv value for %q: %w", key, err)
		}
		switch key {
		case "general.architecture":
			if s, ok := val.(string); ok {
				m.Arch = s
			}
		case m.Arch + ".block_count":
			m.BlockCount = asInt(val)
		case m.Arch + ".embedding_length":
			m.NEmbd = asInt(val)
		case m.Arch + ".context_length":
			m.NContext = asInt(val)
		case m.Arch + ".attention.head_count":
			m.NHeads = asInt(val)
		case m.Arch + ".attention.head_count_kv":
			m.NKVHeads = asInt(val)
		case m.Arch + ".attention.sliding_window":
			w := asInt(val)
			if w > 0 {
				m.SlidingWin = w
			}
		case m.Arch + ".expert_count":
			m.NExperts = asInt(val)
		case m.Arch + ".expert_used_count":
			m.NExpertSel = asInt(val)
		case "general.file_type":
			if v, ok := val.(int64); ok {
				m.FileType = ggmlTypeName(v)
			}
		}
	}

	m.NTensors = int64(nTensors)
	for i := uint64(0); i < nTensors; i++ {
		name, err := readString()
		if err != nil {
			return nil, fmt.Errorf("read tensor name: %w", err)
		}
		ndim, err := u32()
		if err != nil {
			return nil, err
		}
		for j := uint32(0); j < ndim; j++ {
			if _, err := u64(); err != nil {
				return nil, err
			}
		}
		if _, err := u32(); err != nil { // tensor type
			return nil, err
		}
		if _, err := u64(); err != nil { // data offset (present in all versions)
			return nil, err
		}
		if strings.Contains(name, ".nextn.") {
			m.HasNextN = true
		}
		// same check common_speculative_types_from_gguf uses for draft-mtp
		if m.BlockCount > 0 && name == fmt.Sprintf("blk.%d.nextn.eh_proj.weight", m.BlockCount-1) {
			m.MTP = true
		}
	}

	return m, nil
}

var ggmlTypes = map[int64]string{
	0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1",
	8: "q8_0", 9: "q8_1", 10: "q2_K", 11: "q3_K", 12: "q4_K", 13: "q5_K",
	14: "q6_K", 15: "q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
	19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
	29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
	40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
}

func ggmlTypeName(v int64) string {
	if n, ok := ggmlTypes[v]; ok {
		return n
	}
	return fmt.Sprintf("type-%d", v)
}

// kvBytesPerToken returns the KV cache size in bytes for one token at the
// given cache types ("f16", "q8_0", "q4_0"), K and V combined, all layers.
func (m *ggufModel) kvBytesPerToken(ck, cv string) (float64, error) {
	nKV := m.NKVHeads
	if nKV <= 0 {
		nKV = m.NHeads
	}
	if nKV <= 0 || m.NEmbd <= 0 {
		return 0, fmt.Errorf("model has no attention metadata to estimate KV size")
	}
	headDim := m.NEmbd / m.NHeads
	bK, err := kvBytesPerElem(ck)
	if err != nil {
		return 0, err
	}
	bV, err := kvBytesPerElem(cv)
	if err != nil {
		return 0, err
	}
	return float64(m.BlockCount) * float64(nKV) * float64(headDim) * 2 * (bK + bV), nil
}

// bytes per element including block scale overhead (q4_0: 17 bytes per 32 elements)
func kvBytesPerElem(t string) (float64, error) {
	switch t {
	case "f16", "bf16":
		return 2, nil
	case "q8_0":
		return 257 / 256, nil
	case "q4_0":
		return 17 / 16, nil
	case "f32":
		return 4, nil
	default:
		return 0, fmt.Errorf("unsupported cache type %q", t)
	}
}

// kvMiBFor returns the projected KV cache size in MiB for n tokens.
func (m *ggufModel) kvMiBFor(n int, ck, cv string) float64 {
	b, err := m.kvBytesPerToken(ck, cv)
	if err != nil {
		return -1
	}
	return b * float64(n) / (1024 * 1024)
}

func isHFRef(s string) bool {
	// "user/repo" or "user/repo:quant"; local paths are not refs
	if strings.Contains(s, ".gguf") || strings.Contains(s, ".ggmlf") {
		return false
	}
	if strings.ContainsAny(s, " \t") ||
		strings.HasPrefix(s, "/") || strings.HasPrefix(s, "./") ||
		strings.HasPrefix(s, "../") || strings.HasPrefix(s, "~") ||
		strings.Contains(s, ":\\") {
		return false
	}
	base := s
	if i := strings.LastIndex(s, ":"); i > 0 { // strip :quant
		base = s[:i]
	}
	parts := strings.Split(base, "/")
	return len(parts) == 2 && parts[0] != "" && parts[1] != ""
}
