// Shared by the radix_*.comp passes. One pass sorts (key, value) pairs by
// 8 bits of the key (bits [shift, shift + 8)). Keys are split into blocks of
// kBlock = 4 x 256; each block is one workgroup of 256 threads.

#define kRadix 256u
#define kThreads 256u
#define kItems 4u
#define kBlock (kThreads * kItems)

uniform uint count;       // number of pairs
uniform uint shift;       // which 8 bits this pass sorts by
uniform uint numBlocks;

layout(std430, binding = 0) readonly  buffer KeysIn  { uint keysIn[]; };
layout(std430, binding = 1) readonly  buffer ValsIn  { uint valsIn[]; };
layout(std430, binding = 2) writeonly buffer KeysOut { uint keysOut[]; };
layout(std430, binding = 3) writeonly buffer ValsOut { uint valsOut[]; };
// hist[digit * numBlocks + block]: digit-major, so one scan over the whole
// array gives every (digit, block) its first output slot, in order.
layout(std430, binding = 4) buffer Hist { uint hist[]; };

uint Digit(uint key) { return (key >> shift) & (kRadix - 1u); }
