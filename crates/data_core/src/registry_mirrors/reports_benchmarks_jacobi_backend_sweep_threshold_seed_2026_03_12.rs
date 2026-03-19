//! # Jacobi Backend Sweep
//!
//! | family | size | default policy | fastest successful backend | lowest max abs error |
//! | --- | ---: | --- | --- | --- |
//! | clustered_pairs | 4 | reference_f64 | x87 | reference_f64 |
//! | clustered_pairs | 8 | reference_f64 | reference_f64 | x87 |
//! | clustered_pairs | 16 | reference_f64 | reference_f64 | x87 |
//! | clustered_pairs | 24 | reference_f64 | reference_f64 | reference_f64 |
//! | clustered_pairs | 32 | x87 | x87 | reference_f64 |
//! | clustered_pairs | 48 | x87 | x87 | reference_f64 |
//! | clustered_pairs | 64 | x87 | x87 | x87 |
//! | known_spectrum | 4 | reference_f64 | double_double | x87 |
//! | known_spectrum | 8 | reference_f64 | reference_f64 | x87 |
//! | known_spectrum | 16 | reference_f64 | reference_f64 | reference_f64 |
//! | known_spectrum | 24 | reference_f64 | reference_f64 | x87 |
//! | known_spectrum | 32 | x87 | reference_f64 | x87 |
//! | known_spectrum | 48 | x87 | x87 | x87 |
//! | known_spectrum | 64 | x87 | x87 | x87 |
//!
//! ## Rows
//!
//! | family | size | backend | status | selected | median ns | max abs error | rms abs error |
//! | --- | ---: | --- | --- | --- | ---: | ---: | ---: |
//! | known_spectrum | 4 | reference_f64 | ok | true | 340 | 1.776357e-15 | 1.350645e-15 |
//! | known_spectrum | 4 | double_double | ok | false | 320 | 1.776357e-15 | 1.350645e-15 |
//! | known_spectrum | 4 | x87 | ok | false | 360 | 8.881784e-16 | 6.753223e-16 |
//! | known_spectrum | 8 | reference_f64 | ok | true | 8860 | 4.440892e-15 | 1.892271e-15 |
//! | known_spectrum | 8 | double_double | ok | false | 43920 | 3.552714e-15 | 2.613449e-15 |
//! | known_spectrum | 8 | x87 | ok | false | 15890 | 2.664535e-15 | 1.378310e-15 |
//! | known_spectrum | 16 | reference_f64 | ok | true | 87590 | 8.881784e-15 | 3.387659e-15 |
//! | known_spectrum | 16 | double_double | ok | false | 208860 | 1.243450e-14 | 6.519489e-15 |
//! | known_spectrum | 16 | x87 | ok | false | 114050 | 8.881784e-15 | 4.059532e-15 |
//! | known_spectrum | 24 | reference_f64 | ok | true | 377740 | 2.131628e-14 | 9.672861e-15 |
//! | known_spectrum | 24 | double_double | ok | false | 658481 | 3.197442e-14 | 9.222205e-15 |
//! | known_spectrum | 24 | x87 | ok | false | 412190 | 1.776357e-14 | 7.010101e-15 |
//! | known_spectrum | 32 | reference_f64 | ok | false | 1253252 | 4.263256e-14 | 1.430148e-14 |
//! | known_spectrum | 32 | double_double | ok | false | 1829053 | 6.394885e-14 | 2.774372e-14 |
//! | known_spectrum | 32 | x87 | ok | true | 1265632 | 3.197442e-14 | 1.345347e-14 |
//! | known_spectrum | 48 | reference_f64 | ok | false | 6150161 | 1.278977e-13 | 3.429182e-14 |
//! | known_spectrum | 48 | double_double | ok | false | 7621622 | 1.136868e-13 | 4.021394e-14 |
//! | known_spectrum | 48 | x87 | ok | true | 5263619 | 7.105427e-14 | 2.752753e-14 |
//! | known_spectrum | 64 | reference_f64 | ok | false | 18808791 | 1.634248e-13 | 4.144245e-14 |
//! | known_spectrum | 64 | double_double | ok | false | 22177027 | 2.771117e-13 | 7.932885e-14 |
//! | known_spectrum | 64 | x87 | ok | true | 16580678 | 1.207923e-13 | 3.020164e-14 |
//! | clustered_pairs | 4 | reference_f64 | ok | true | 350 | 8.881784e-16 | 8.382000e-16 |
//! | clustered_pairs | 4 | double_double | ok | false | 320 | 8.881784e-16 | 8.382000e-16 |
//! | clustered_pairs | 4 | x87 | ok | false | 290 | 8.881784e-16 | 7.021667e-16 |
//! | clustered_pairs | 8 | reference_f64 | ok | true | 5830 | 1.332268e-15 | 5.495324e-16 |
//! | clustered_pairs | 8 | double_double | ok | false | 33010 | 1.776357e-15 | 9.096040e-16 |
//! | clustered_pairs | 8 | x87 | ok | false | 9820 | 4.440892e-16 | 1.798767e-16 |
//! | clustered_pairs | 16 | reference_f64 | ok | true | 80900 | 6.217249e-15 | 2.488268e-15 |
//! | clustered_pairs | 16 | double_double | ok | false | 201540 | 5.329071e-15 | 2.846266e-15 |
//! | clustered_pairs | 16 | x87 | ok | false | 105020 | 2.664535e-15 | 1.370181e-15 |
//! | clustered_pairs | 24 | reference_f64 | ok | true | 381731 | 7.105427e-15 | 3.388151e-15 |
//! | clustered_pairs | 24 | double_double | ok | false | 634261 | 2.220446e-14 | 8.478530e-15 |
//! | clustered_pairs | 24 | x87 | ok | false | 387411 | 8.881784e-15 | 3.590460e-15 |
//! | clustered_pairs | 32 | reference_f64 | ok | false | 1269832 | 2.131628e-14 | 7.970539e-15 |
//! | clustered_pairs | 32 | double_double | ok | false | 1797093 | 5.329071e-14 | 1.886756e-14 |
//! | clustered_pairs | 32 | x87 | ok | true | 1131402 | 2.131628e-14 | 6.865901e-15 |
//! | clustered_pairs | 48 | reference_f64 | ok | false | 6180980 | 4.973799e-14 | 1.815198e-14 |
//! | clustered_pairs | 48 | double_double | ok | false | 7000202 | 2.451372e-13 | 5.772514e-14 |
//! | clustered_pairs | 48 | x87 | ok | true | 4823308 | 7.105427e-14 | 1.643168e-14 |
//! | clustered_pairs | 64 | reference_f64 | ok | false | 19071942 | 9.947598e-14 | 2.212287e-14 |
//! | clustered_pairs | 64 | double_double | ok | false | 22784787 | 1.865175e-13 | 5.511862e-14 |
//! | clustered_pairs | 64 | x87 | ok | true | 15063755 | 6.750156e-14 | 2.105763e-14 |
//!
