#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace featpack {

constexpr std::size_t kFeatures = 29;
constexpr std::size_t kBytes = kFeatures * 4;

static_assert(sizeof(float) == 4, "float must be 32-bit");
static_assert(std::numeric_limits<float>::is_iec559, "float must be IEEE-754");

inline std::uint32_t float_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

inline float bits_float(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

// Value-based shifts: the result does not depend on host byte order.
inline void put_be32(std::uint8_t* p, std::uint32_t v) {
    p[0] = static_cast<std::uint8_t>(v >> 24);
    p[1] = static_cast<std::uint8_t>(v >> 16);
    p[2] = static_cast<std::uint8_t>(v >> 8);
    p[3] = static_cast<std::uint8_t>(v);
}

inline std::uint32_t get_be32(const std::uint8_t* p) {
    return (static_cast<std::uint32_t>(p[0]) << 24) |
           (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) |
           static_cast<std::uint32_t>(p[3]);
}

// Packs kFeatures floats into kBytes bytes, big-endian, bit patterns preserved.
inline void pack(const float* in, std::uint8_t* out) {
    for (std::size_t i = 0; i < kFeatures; ++i) put_be32(out + 4 * i, float_bits(in[i]));
}

inline void unpack(const std::uint8_t* in, float* out) {
    for (std::size_t i = 0; i < kFeatures; ++i) out[i] = bits_float(get_be32(in + 4 * i));
}

inline bool host_is_little_endian() {
    const std::uint32_t one = 1;
    std::uint8_t b;
    std::memcpy(&b, &one, 1);
    return b == 1;
}

}  // namespace featpack
