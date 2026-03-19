//! # Jacobi Backend Sweep
//!
//! | family | size | default policy | fastest successful backend | lowest max abs error |
//! | --- | ---: | --- | --- | --- |
//! | clustered_pairs | 60 | x87 | x87 | reference_f64 |
//! | clustered_pairs | 64 | x87 | x87 | x87 |
//! | clustered_pairs | 68 | x87 | x87 | x87 |
//! | geometric_decay | 60 | x87 | reference_f64 | x87 |
//! | geometric_decay | 64 | x87 | reference_f64 | x87 |
//! | geometric_decay | 68 | x87 | reference_f64 | x87 |
//! | known_spectrum | 60 | x87 | reference_f64 | reference_f64 |
//! | known_spectrum | 64 | x87 | reference_f64 | x87 |
//! | known_spectrum | 68 | x87 | reference_f64 | x87 |
//! | spiked_tail | 60 | x87 | reference_f64 | x87 |
//! | spiked_tail | 64 | x87 | reference_f64 | x87 |
//! | spiked_tail | 68 | x87 | reference_f64 | x87 |
//!
//! ## Rows
//!
//! | family | size | backend | status | selected | median ns | max abs error | rms abs error |
//! | --- | ---: | --- | --- | --- | ---: | ---: | ---: |
//! | known_spectrum | 60 | reference_f64 | ok | false | 6938436 | 9.947598e-14 | 3.754959e-14 |
//! | known_spectrum | 60 | double_double | ok | false | 9167324 | 1.492140e-13 | 6.261279e-14 |
//! | known_spectrum | 60 | x87 | ok | true | 7080959 | 1.278977e-13 | 4.111469e-14 |
//! | known_spectrum | 64 | reference_f64 | ok | false | 8883191 | 1.634248e-13 | 4.144245e-14 |
//! | known_spectrum | 64 | double_double | ok | false | 11520384 | 2.771117e-13 | 7.932885e-14 |
//! | known_spectrum | 64 | x87 | ok | true | 9030903 | 1.207923e-13 | 3.020164e-14 |
//! | known_spectrum | 68 | reference_f64 | ok | false | 11342442 | 2.629008e-13 | 6.371878e-14 |
//! | known_spectrum | 68 | double_double | ok | false | 14388870 | 3.694822e-13 | 1.110576e-13 |
//! | known_spectrum | 68 | x87 | ok | true | 11441703 | 1.705303e-13 | 4.436568e-14 |
//! | clustered_pairs | 60 | reference_f64 | ok | false | 7061469 | 6.750156e-14 | 2.222048e-14 |
//! | clustered_pairs | 60 | double_double | ok | false | 9067713 | 2.238210e-13 | 6.133761e-14 |
//! | clustered_pairs | 60 | x87 | ok | true | 6950817 | 1.278977e-13 | 2.768077e-14 |
//! | clustered_pairs | 64 | reference_f64 | ok | false | 9192175 | 9.947598e-14 | 2.212287e-14 |
//! | clustered_pairs | 64 | double_double | ok | false | 11521283 | 1.865175e-13 | 5.511862e-14 |
//! | clustered_pairs | 64 | x87 | ok | true | 8876881 | 6.750156e-14 | 2.105763e-14 |
//! | clustered_pairs | 68 | reference_f64 | ok | false | 11648995 | 6.750156e-14 | 2.871491e-14 |
//! | clustered_pairs | 68 | double_double | ok | false | 14357949 | 3.019807e-13 | 8.535050e-14 |
//! | clustered_pairs | 68 | x87 | ok | true | 11368841 | 5.684342e-14 | 1.760492e-14 |
//! | geometric_decay | 60 | reference_f64 | ok | false | 8505276 | 1.221245e-15 | 2.409359e-16 |
//! | geometric_decay | 60 | double_double | ok | false | 11840248 | 2.997602e-15 | 4.261439e-16 |
//! | geometric_decay | 60 | x87 | ok | true | 9162094 | 4.996004e-16 | 1.117374e-16 |
//! | geometric_decay | 64 | reference_f64 | ok | false | 11202859 | 1.776357e-15 | 3.162526e-16 |
//! | geometric_decay | 64 | double_double | ok | false | 15458912 | 2.886580e-15 | 4.902476e-16 |
//! | geometric_decay | 64 | x87 | ok | true | 11855007 | 9.992007e-16 | 1.541150e-16 |
//! | geometric_decay | 68 | reference_f64 | ok | false | 13767562 | 2.309502e-14 | 3.534293e-15 |
//! | geometric_decay | 68 | double_double | ok | false | 18746934 | 6.217249e-15 | 1.087245e-15 |
//! | geometric_decay | 68 | x87 | ok | true | 14594022 | 1.221245e-15 | 1.635451e-16 |
//! | spiked_tail | 60 | reference_f64 | ok | false | 6398160 | 5.684342e-14 | 7.810854e-15 |
//! | spiked_tail | 60 | double_double | ok | false | 8825830 | 6.394885e-14 | 1.066081e-14 |
//! | spiked_tail | 60 | x87 | ok | true | 6829195 | 3.197442e-14 | 4.134564e-15 |
//! | spiked_tail | 64 | reference_f64 | ok | false | 8282663 | 1.989520e-13 | 2.525956e-14 |
//! | spiked_tail | 64 | double_double | ok | false | 11120009 | 2.700062e-13 | 3.704439e-14 |
//! | spiked_tail | 64 | x87 | ok | true | 8678058 | 4.263256e-14 | 5.962483e-15 |
//! | spiked_tail | 68 | reference_f64 | ok | false | 10573722 | 9.947598e-14 | 1.539586e-14 |
//! | spiked_tail | 68 | double_double | ok | false | 13946414 | 1.989520e-13 | 2.500313e-14 |
//! | spiked_tail | 68 | x87 | ok | true | 10977267 | 4.263256e-14 | 6.519903e-15 |
//!
