/* Correspondence harness for the extracted DP4 operand-admission gate.
 *
 * dp4_operand_admit B = (0 <=? B) && (4*B*B <=? 131072); the proven boundary
 * is B = 181 admitted, B = 182 refused (dp4_admit_boundary).  The harness
 * encodes each B through the glue, applies the extracted closure, decodes the
 * bool, and compares with the native predicate.  Known-good rows: 0, 1, 127,
 * 128, 181.  Known-bad rows: -1, 182, 1000, 131072.  Exit 0 on agreement.
 */
#include <stdio.h>
#include <stdlib.h>
#include <gc_stack.h>
#include "r2vb_dp4_glue.h"

extern value body(struct thread_info *);

static value make_pos(struct thread_info *ti, unsigned long long n)
{
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

static int ref_admit(long long b)
{
   return b >= 0 && 4 * b * b <= 131072;
}

int main(void)
{
   struct thread_info *ti = make_tinfo();
   static const long long fx[] = {0, 1, 127, 128, 181, -1, 182, 1000, 131072};
   unsigned nfail = 0;
   for (unsigned i = 0; i < sizeof fx / sizeof fx[0]; i++) {
      value clo = body(ti);
      value r = call(ti, clo, make_Z(ti, fx[i]));
      /* bool tags observed via glue: true = 1, false = 0. */
      int got = get_Corelib_Init_Datatypes_bool_tag(r) == 1;
      int want = ref_admit(fx[i]);
      printf("B=%lld got=%d want=%d %s\n", fx[i], got, want,
             got == want ? "MATCH" : "MISMATCH");
      if (got != want)
         nfail++;
   }
   printf("dp4_operand_admit correspondence: %u failure(s)\n", nfail);
   return nfail ? EXIT_FAILURE : EXIT_SUCCESS;
}
