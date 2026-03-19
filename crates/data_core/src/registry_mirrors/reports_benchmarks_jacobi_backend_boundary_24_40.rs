//! # Jacobi Backend Sweep
//!
//! | family | size | default policy | fastest successful backend | lowest max abs error |
//! | --- | ---: | --- | --- | --- |
//! | clustered_pairs | 24 | reference_f64 | reference_f64 | reference_f64 |
//! | clustered_pairs | 28 | reference_f64 | reference_f64 | x87 |
//! | clustered_pairs | 32 | reference_f64 | reference_f64 | reference_f64 |
//! | clustered_pairs | 36 | reference_f64 | reference_f64 | x87 |
//! | clustered_pairs | 40 | reference_f64 | reference_f64 | reference_f64 |
//! | geometric_decay | 24 | reference_f64 | reference_f64 | x87 |
//! | geometric_decay | 28 | reference_f64 | reference_f64 | x87 |
//! | geometric_decay | 32 | reference_f64 | reference_f64 | x87 |
//! | geometric_decay | 36 | reference_f64 | reference_f64 | x87 |
//! | geometric_decay | 40 | reference_f64 | reference_f64 | double_double |
//! | known_spectrum | 24 | reference_f64 | reference_f64 | x87 |
//! | known_spectrum | 28 | reference_f64 | reference_f64 | x87 |
//! | known_spectrum | 32 | reference_f64 | reference_f64 | x87 |
//! | known_spectrum | 36 | reference_f64 | reference_f64 | reference_f64 |
//! | known_spectrum | 40 | reference_f64 | reference_f64 | x87 |
//! | spiked_tail | 24 | reference_f64 | reference_f64 | x87 |
//! | spiked_tail | 28 | reference_f64 | reference_f64 | x87 |
//! | spiked_tail | 32 | reference_f64 | reference_f64 | reference_f64 |
//! | spiked_tail | 36 | reference_f64 | reference_f64 | x87 |
//! | spiked_tail | 40 | reference_f64 | reference_f64 | reference_f64 |
//!
//! ## Rows
//!
//! | family | size | backend | status | selected | median ns | max abs error | rms abs error |
//! | --- | ---: | --- | --- | --- | ---: | ---: | ---: |
//! | known_spectrum | 24 | reference_f64 | ok | true | 191793 | 2.131628e-14 | 9.672861e-15 |
//! | known_spectrum | 24 | double_double | ok | false | 454966 | 3.197442e-14 | 9.222205e-15 |
//! | known_spectrum | 24 | x87 | ok | false | 249823 | 1.776357e-14 | 7.010101e-15 |
//! | known_spectrum | 28 | reference_f64 | ok | true | 330894 | 4.973799e-14 | 1.668424e-14 |
//! | known_spectrum | 28 | double_double | ok | false | 692569 | 3.730349e-14 | 1.741237e-14 |
//! | known_spectrum | 28 | x87 | ok | false | 408065 | 2.131628e-14 | 8.201197e-15 |
//! | known_spectrum | 32 | reference_f64 | ok | true | 616827 | 4.263256e-14 | 1.430148e-14 |
//! | known_spectrum | 32 | double_double | ok | false | 1189855 | 6.394885e-14 | 2.774372e-14 |
//! | known_spectrum | 32 | x87 | ok | false | 734599 | 3.197442e-14 | 1.345347e-14 |
//! | known_spectrum | 36 | reference_f64 | ok | true | 958092 | 5.684342e-14 | 1.845243e-14 |
//! | known_spectrum | 36 | double_double | ok | false | 1688071 | 8.171241e-14 | 3.096169e-14 |
//! | known_spectrum | 36 | x87 | ok | false | 1101963 | 5.684342e-14 | 1.983491e-14 |
//! | known_spectrum | 40 | reference_f64 | ok | true | 1414548 | 7.105427e-14 | 2.163588e-14 |
//! | known_spectrum | 40 | double_double | ok | false | 2319249 | 7.815970e-14 | 3.437321e-14 |
//! | known_spectrum | 40 | x87 | ok | false | 1593360 | 4.973799e-14 | 1.729986e-14 |
//! | clustered_pairs | 24 | reference_f64 | ok | true | 182542 | 7.105427e-15 | 3.388151e-15 |
//! | clustered_pairs | 24 | double_double | ok | false | 427646 | 2.220446e-14 | 8.478530e-15 |
//! | clustered_pairs | 24 | x87 | ok | false | 232212 | 8.881784e-15 | 3.590460e-15 |
//! | clustered_pairs | 28 | reference_f64 | ok | true | 310014 | 2.486900e-14 | 6.504727e-15 |
//! | clustered_pairs | 28 | double_double | ok | false | 644948 | 3.730349e-14 | 1.356005e-14 |
//! | clustered_pairs | 28 | x87 | ok | false | 380425 | 1.065814e-14 | 4.154922e-15 |
//! | clustered_pairs | 32 | reference_f64 | ok | true | 588968 | 2.131628e-14 | 7.970539e-15 |
//! | clustered_pairs | 32 | double_double | ok | false | 1120514 | 5.329071e-14 | 1.886756e-14 |
//! | clustered_pairs | 32 | x87 | ok | false | 696519 | 2.131628e-14 | 6.865901e-15 |
//! | clustered_pairs | 36 | reference_f64 | ok | true | 967872 | 3.019807e-14 | 1.132947e-14 |
//! | clustered_pairs | 36 | double_double | ok | false | 1660841 | 4.085621e-14 | 1.680534e-14 |
//! | clustered_pairs | 36 | x87 | ok | false | 1075204 | 2.486900e-14 | 7.695500e-15 |
//! | clustered_pairs | 40 | reference_f64 | ok | true | 1445818 | 3.197442e-14 | 1.105513e-14 |
//! | clustered_pairs | 40 | double_double | ok | false | 2294849 | 1.243450e-13 | 3.520274e-14 |
//! | clustered_pairs | 40 | x87 | ok | false | 1554759 | 4.263256e-14 | 1.193089e-14 |
//! | geometric_decay | 24 | reference_f64 | ok | true | 248963 | 6.661338e-16 | 1.648845e-16 |
//! | geometric_decay | 24 | double_double | ok | false | 661538 | 1.665335e-15 | 3.724160e-16 |
//! | geometric_decay | 24 | x87 | ok | false | 335824 | 2.220446e-16 | 6.791319e-17 |
//! | geometric_decay | 28 | reference_f64 | ok | true | 424135 | 7.771561e-16 | 1.571596e-16 |
//! | geometric_decay | 28 | double_double | ok | false | 1002392 | 1.332268e-15 | 3.769007e-16 |
//! | geometric_decay | 28 | x87 | ok | false | 544397 | 4.440892e-16 | 1.105289e-16 |
//! | geometric_decay | 32 | reference_f64 | ok | true | 737169 | 1.221245e-15 | 3.034922e-16 |
//! | geometric_decay | 32 | double_double | ok | false | 1528189 | 1.466709e-15 | 4.329993e-16 |
//! | geometric_decay | 32 | x87 | ok | false | 886871 | 9.992007e-16 | 2.043664e-16 |
//! | geometric_decay | 36 | reference_f64 | ok | true | 1267966 | 1.443290e-15 | 2.725542e-16 |
//! | geometric_decay | 36 | double_double | ok | false | 2222337 | 1.110223e-15 | 2.751588e-16 |
//! | geometric_decay | 36 | x87 | ok | false | 1370607 | 9.436896e-16 | 2.228736e-16 |
//! | geometric_decay | 40 | reference_f64 | ok | true | 1847473 | 2.553513e-15 | 4.351397e-16 |
//! | geometric_decay | 40 | double_double | ok | false | 3203270 | 8.326673e-16 | 1.645701e-16 |
//! | geometric_decay | 40 | x87 | ok | false | 2074166 | 1.332268e-15 | 2.315673e-16 |
//! | spiked_tail | 24 | reference_f64 | ok | true | 207232 | 6.394885e-14 | 1.380273e-14 |
//! | spiked_tail | 24 | double_double | ok | false | 537887 | 4.973799e-14 | 1.025868e-14 |
//! | spiked_tail | 24 | x87 | ok | false | 283054 | 2.131628e-14 | 4.500357e-15 |
//! | spiked_tail | 28 | reference_f64 | ok | true | 353974 | 1.421085e-14 | 3.557382e-15 |
//! | spiked_tail | 28 | double_double | ok | false | 810510 | 1.776357e-14 | 3.865146e-15 |
//! | spiked_tail | 28 | x87 | ok | false | 462296 | 1.243450e-14 | 2.357028e-15 |
//! | spiked_tail | 32 | reference_f64 | ok | true | 566197 | 7.105427e-15 | 1.691892e-15 |
//! | spiked_tail | 32 | double_double | ok | false | 1136414 | 3.197442e-14 | 6.599453e-15 |
//! | spiked_tail | 32 | x87 | ok | false | 710519 | 1.065814e-14 | 2.665251e-15 |
//! | spiked_tail | 36 | reference_f64 | ok | true | 920021 | 4.263256e-14 | 8.960628e-15 |
//! | spiked_tail | 36 | double_double | ok | false | 1705932 | 6.394885e-14 | 1.308047e-14 |
//! | spiked_tail | 36 | x87 | ok | false | 1090223 | 2.842171e-14 | 5.296406e-15 |
//! | spiked_tail | 40 | reference_f64 | ok | true | 1384097 | 2.842171e-14 | 6.509361e-15 |
//! | spiked_tail | 40 | double_double | ok | false | 2339749 | 3.552714e-14 | 8.449681e-15 |
//! | spiked_tail | 40 | x87 | ok | false | 1590980 | 4.263256e-14 | 6.971338e-15 |
//!
