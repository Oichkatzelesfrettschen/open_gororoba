/* Correspondence harness for the extracted R2VB DP4 lane.
 *
 * The extracted dp4 (OpenGororoba.R2VBTransformDP4.dp4.c) computes the 4-term
 * integer dot product over CertiCoq-encoded Z; this harness encodes each
 * fixture through the generated glue constructors, applies the closure one
 * argument at a time through call(), decodes the resulting Z, and compares it
 * with a native long long reference.  Fixtures cover zero, unit, mixed-sign,
 * the admitted boundary magnitude 181, the first refused magnitude 182, and
 * values far outside the FP24 window, so the integer semantics is checked
 * independently of the admission gate.  Exit 0 on full agreement; any
 * disagreement names the fixture and returns 1.
 */
#include <stdio.h>
#include <stdlib.h>
#include <gc_stack.h>
#include "r2vb_dp4_glue.h"

extern value body(struct thread_info *);

static value make_pos(struct thread_info *ti, unsigned long long n)
{
   /* Build a positive (n >= 1) msb-first: xH for the leading 1, then xO/xI. */
   int bits[64], k = 0;
   while (n > 1) { bits[k++] = (int)(n & 1); n >>= 1; }
   value p = make_Corelib_Numbers_BinNums_positive_xH();
   while (k > 0) {
      k--;
      p = bits[k] ? alloc_make_Corelib_Numbers_BinNums_positive_xI(ti, p)
                  : alloc_make_Corelib_Numbers_BinNums_positive_xO(ti, p);
   }
   return p;
}

static value make_Z(struct thread_info *ti, long long n)
{
   if (n == 0)
      return make_Corelib_Numbers_BinNums_Z_Z0();
   if (n > 0)
      return alloc_make_Corelib_Numbers_BinNums_Z_Zpos(ti, make_pos(ti, (unsigned long long)n));
   return alloc_make_Corelib_Numbers_BinNums_Z_Zneg(ti, make_pos(ti, (unsigned long long)(-n)));
}

static unsigned long long decode_pos(value p)
{
   /* glue tags: xI = 0, xO = 1, xH = 2 (constructor order in BinNums). */
   unsigned long long n = 0, bit = 1;
   unsigned long long acc = 0;
   int shift = 0;
   for (;;) {
      unsigned long long tag = get_Corelib_Numbers_BinNums_positive_tag(p);
      if (tag == 2) { acc |= (1ULL << shift); return acc; }
      if (tag == 0) acc |= (1ULL << shift);
      shift++;
      p = get_args(p)[0];
   }
   (void)n; (void)bit;
}

static long long decode_Z(value z)
{
   /* glue tags: Z0 = 0, Zpos = 1, Zneg = 2. */
   unsigned long long tag = get_Corelib_Numbers_BinNums_Z_tag(z);
   if (tag == 0) return 0;
   long long m = (long long)decode_pos(get_args(z)[0]);
   return tag == 1 ? m : -m;
}

static long long ref_dp4(const long long *a, const long long *b)
{
   return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
}

int main(void)
{
   struct thread_info *ti = make_tinfo();
   static const long long fx[][8] = {
      {0, 0, 0, 0, 0, 0, 0, 0},
      {1, 0, 0, 0, 1, 0, 0, 0},
      {1, 2, 3, 4, 5, 6, 7, 8},
      {-1, 2, -3, 4, 5, -6, 7, -8},
      {181, 181, 181, 181, 181, 181, 181, 181},
      {-181, 181, -181, 181, 181, -181, 181, -181},
      {182, 182, 182, 182, 182, 182, 182, 182},
      {1000, -2000, 3000, -4000, 5000, 6000, -7000, 8000},
      {131072, 1, -131072, 1, 1, 131072, 1, -131072},
   };
   unsigned nfail = 0;
   for (unsigned i = 0; i < sizeof fx / sizeof fx[0]; i++) {
      value clo = body(ti);
      for (int k = 0; k < 8; k++)
         clo = call(ti, clo, make_Z(ti, fx[i][k]));
      long long got = decode_Z(clo);
      long long want = ref_dp4(fx[i], fx[i] + 4);
      printf("fixture=%u got=%lld want=%lld %s\n", i, got, want,
             got == want ? "MATCH" : "MISMATCH");
      if (got != want)
         nfail++;
   }
   printf("dp4 correspondence: %u failure(s)\n", nfail);
   return nfail ? EXIT_FAILURE : EXIT_SUCCESS;
}
