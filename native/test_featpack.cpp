#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "featpack.h"

using namespace featpack;

static int failures = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

static bool bytes_eq(const std::uint8_t* a, const std::uint8_t* b, std::size_t n) {
    return std::memcmp(a, b, n) == 0;
}

int main() {
    std::printf("host byte order: %s\n", host_is_little_endian() ? "little" : "big");

    // Golden vectors: IEEE-754 single precision, big-endian layout.
    struct Golden { float v; std::uint8_t be[4]; };
    const Golden golden[] = {
        {1.0f,  {0x3F, 0x80, 0x00, 0x00}},
        {-2.5f, {0xC0, 0x20, 0x00, 0x00}},
        {0.1f,  {0x3D, 0xCC, 0xCC, 0xCD}},
        {0.0f,  {0x00, 0x00, 0x00, 0x00}},
    };
    for (const Golden& g : golden) {
        std::uint8_t b[4];
        put_be32(b, float_bits(g.v));
        CHECK(bytes_eq(b, g.be, 4));
    }

    // Round trip must be bit-exact, including special values.
    float in[kFeatures];
    for (std::size_t i = 0; i < kFeatures; ++i) in[i] = static_cast<float>(i) * 0.37f - 5.0f;
    in[0] = -0.0f;
    in[1] = bits_float(0x7F800000u);   // +inf
    in[2] = bits_float(0xFF800000u);   // -inf
    in[3] = bits_float(0x00000001u);   // smallest denormal
    in[4] = bits_float(0x7FC00000u);   // quiet NaN
    in[5] = bits_float(0x7F7FFFFFu);   // largest finite

    std::uint8_t buf[kBytes];
    float out[kFeatures];
    pack(in, buf);
    unpack(buf, out);
    for (std::size_t i = 0; i < kFeatures; ++i) CHECK(float_bits(in[i]) == float_bits(out[i]));

    // Layout is fixed: first float's first byte is its sign/exponent byte (-0.0f -> 0x80).
    CHECK(buf[0] == 0x80 && buf[1] == 0x00 && buf[2] == 0x00 && buf[3] == 0x00);
    CHECK(buf[4] == 0x7F && buf[5] == 0x80);   // +inf

    // The raw in-memory bytes of 1.0f differ between hosts; the packed bytes must not.
    const float one = 1.0f;
    std::uint8_t raw[4];
    std::memcpy(raw, &one, 4);
    std::uint8_t packed_one[4];
    put_be32(packed_one, float_bits(one));
    const std::uint8_t want[4] = {0x3F, 0x80, 0x00, 0x00};
    CHECK(bytes_eq(packed_one, want, 4));
    CHECK(bytes_eq(raw, want, 4) == !host_is_little_endian());
    std::printf("raw bytes of 1.0f in memory : %02X %02X %02X %02X\n", raw[0], raw[1], raw[2], raw[3]);
    std::printf("packed (big-endian) 1.0f    : %02X %02X %02X %02X\n",
                packed_one[0], packed_one[1], packed_one[2], packed_one[3]);

    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("all featpack checks passed\n");
    return EXIT_SUCCESS;
}
