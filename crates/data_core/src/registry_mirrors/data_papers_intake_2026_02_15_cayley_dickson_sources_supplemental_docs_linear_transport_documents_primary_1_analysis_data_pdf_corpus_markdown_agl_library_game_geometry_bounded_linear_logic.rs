//! # Extracted text: bounded_linear_logic.pdf
//!
//! - source_root: `/home/eirikr/Documents/AGL_Library/Game_Geometry_Documentation`
//! - source_relpath: `bounded_linear_logic.pdf`
//! - source_abs: `/home/eirikr/Documents/AGL_Library/Game_Geometry_Documentation/bounded_linear_logic.pdf`
//! - detected_kind: `pdf`
//! - extracted_at_utc: `2026-01-02T17:30:58+00:00`
//! - pages: `10`
//! - title: ``
//! - author: `IJHT`
//! - subject: ``
//! - keywords: ``
//! - creation_date: `Wed Mar 31 01:48:35 2021 PDT`
//! - mod_date: `Wed Mar 31 01:48:35 2021 PDT`
//! - encrypted: `no`
//!
//! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~text
//! Advances in Modelling and Analysis B
//! Vol.64, No.1, March, 2021, pp. **-**
//! Journal homepage: http://iieta.org/journals/AMA_B
//!
//! Cascade-Forward Neural Network Based on Resilient Backpropagation for Simultaneous
//! Parameters and State Space Estimations of Brushed DC Machines
//! Hacene MELLAH 1,2* , Kamel Eddine HEMSAS2 , Rachid TALEB 3
//! 1
//!
//! Electrical engineering department, faculty of sciences and applied sciences, University of Akli Mouhand Oulhadj-Bouira,
//! Algeria
//! 2
//! Department of Electrical Engineering, Faculty of Technology, Ferhat Abbas Setif 1 University, LAS laboratory, Setif, Algeria
//! 3 Department of Electrical Engineering, LGEER Laboratory, Hassiba Benbouali University, Chlef, Algeria
//! Corresponding Author Email: has.mel@gmail.com
//! https://doi.org/10.18280/ria.xxxxxx
//!
//! ABSTRACT
//!
//! Received:
//! Accepted:
//!
//! A sensorless speed, average temperature and resistance estimation technique based on Neural
//! Network (NN) for brushed DC machines is proposed in this paper. The literature on parameters
//! and state spaces estimations of the Brushed DC machines, shows a variety of approaches.
//! However, these observers are sensitive to a noise, on the model accuracy also are difficult to
//! stabilize and to converge. Furthermore, the majority of earlier works, estimate either the speed
//! or the temperature or the winding resistance. According to the literatures, the Resilient
//! backpropagation (RBP) as is the known as the faster BP algorithm, Cascade-Forward Neural
//! Network (CFNN), is known as the among accelerated learning backpropagation algorithms ,
//! that's why where it is found in several researches, also in several applications in these few
//! years. The main objective of this paper is to introduce an intelligent sensor based on resilient
//! BP to estimate simultaneously the speed, armature temperature and resistance of brushed DC
//! machines only from the measured current and voltage. A comparison between the obtained
//! results and the results of traditional estimator has been made to prove the ability of the
//! proposed method. This method can be embedded in thermal monitoring systems, in high
//! performance motor drives.
//!
//! Keywords:
//! Parameters and state-space estimations,
//! Cascade-Forward Neural Network,
//! Resilient
//! backpropagation,
//! speed
//! estimation,
//! temperature estimation,
//! resistance estimation, Brushed DC
//! electric motor
//!
//! 1. INTRODUCTION
//! In the last few years there has been a growing interest in
//! thermal aspects of electrical machines and their effects, the
//! electrical and mechanical time constants varied for each
//! temperature variation, also the electrical resistance and its
//! back EMF depend on temperature [1]; during operation, the
//! chara cteristics, performance of electric motors were not the
//! same as those the design's [2], as a result, the temperature
//! quantification is very important to the best control and the
//! reliability of electrical machines.
//! The normal effect of thermal aging is to make the insulation
//! system vulnerable to other factors and effects that currently
//! produce failures [3, 4]. Once the insulation loses its physical
//! performance, it can no longer withstand the various dielectric,
//! mechanical and environmental effects, because of these
//! catastrophic effects many researchers interested in the
//! insulation systems monitoring methods of electrical machines
//! [5]. Among the causes of thermal faults are: overloads [6],
//! cyclic mode [7], over voltage and unbalances voltage [8],
//! distortion voltage [4], thermal insulation aging [3], obstructed
//! or impaired cooling [9], poor design and manufacture [3], skin
//! effect [10], the interested reader is referred to [3-10] for more
//! detailed about the cause of stator and rotor failures.
//! For several years, great effort has been devoted to the
//! temperature and speed measurement of electrical machines, in
//! literature, we find several methods about temperature [11 -13]
//! and speed measurements [14] of electrical apparatus. The
//! direct measurement of temperature in electric DC machines is
//!
//! an old theme treated at their time with less pressure [13, 15,
//! 16], on the other hand, (indirect) some author obtained the
//! average winding temperature from the resistance measurement
//! [13], a more modern method can be found in [12, 17, 18], but
//! measurement of the temperature poses two major problems:
//! the measurement point i.e the optimum sensor placement and
//! the obtaining of the thermal information from the rotor [18],
//! in the same manner for the speed measurement, some
//! difficulties are presented her [19].
//! Moreover, obtaining information from sensors installed on
//! the armature adds techno-economics difficulties on the
//! measurement chain; these technical and economic
//! disadvantages of physical sensors as well known to
//! researchers, pushes them for sensorless solutions [17, 20, 21].
//! To solve the problem of sensorless speed estimation, many
//! researchers have proposed various methods [22, 23], a
//! position-sensorless control of brushless DC motor for electric
//! vehicles application is presented in [24], a low-cost lowresolution sensorless for brushed DC motor is proposed and
//! experimentally validated in [25], a sensorless estimation based
//! on support vector machines is proposed by [26]. however, [27]
//! suggest a speed estimation based quantized sensors of PMDC
//! motors. An excellent review about position-sensorless
//! operation of brushless permanent-magnet machines is
//! presented in [22].
//! One of the first examples of temperature estimation is
//! presented in [28], when the authors apply a Luenberger
//! observer both for DC rolling mill motor and a squirrel cage
//! induction motor, another solution is described in [29] where
//!
//!
//! --- PAGE BREAK ---
//! the authors use a steady-state EKF associated with its transient
//! version, nevertheless , for the resistance estimation some
//! author combine between EKF with the smooth variable
//! structure filter [30].
//! Some research on bi-estimation has been done [31, 32], in
//! our point view the most interesting approach to this issue has
//! been proposed by Acarnley et al. in [32], where they propose,
//! applies and experimentally validated the transient EKF to
//! estimate the speed and armature temperature in a brushed DC
//! motor. However, we can summarize the EKF limitations for
//! three points, if the system is incorrectly modelled the filter
//! could quickly diverge, the EKF assumes that the noises are
//! Gaussian [33-35] may not be the reality [36] and eventually, if
//! the initial state estimate values are incorrect also the filter may
//! diverge [37]. Furthermore, using an EKF, which is difficult to
//! stabilize with the sensible choices of covariance matrices [34 36].
//! However, to the authors' knowledge, very few publications
//! can be found in the literature dealing with the simultaneous
//! estimation of speed, armature temperature of brushed DC
//! machines [32], especially by intelligent estimators based on
//! NN [38], despite the NN has been applied to process control
//! [39], diagnostics, identification [40], prediction [41], power
//! electronics [42] and robotics [43], social studies [44], building
//! [45] and medical [46].
//! In the paper [38], the authors discuss how to avoid the limits
//! of the standard NN based on Multilayer Perceptron with
//! Levenberg-Marquardt Backpropagation in their application,
//! and propose as a solution a CFNN based on Bayesian
//! Regulation backpropagation (BRBP). However, an NN based
//! on BRBP is very accurate but need an enormous time to
//! converge and is known as a slow algorithm to converge [44,
//! 45], based on the approach presented in [38], the purpose of
//! this paper is interest to a CFNN based on fast learning
//! algorithm. According to the literatures the Resilient
//! backpropagation (RBP) as is the known as the faster BP [4650], the main objective of this paper is to introduce an
//! intelligent NN-based resilient BP sensor to estimate
//! simultaneously the speed, armature temperature and resistance
//! only from the measured current and voltage.
//! The remainder of the paper is organized as follows sections:
//! Section II describes the thermal model of Brushed DC motor;
//! Section III discusses on the NN and CFNN based on RBP and
//! give some detail about RP properties and its variants.
//! Simulation results are presented, commented and compared
//! with the earlier results in Section IV; Section V concludes the
//! paper.
//! 2. THERMAL MODEL OF BRUSHED DC MOTOR
//! The researchers begin to interest to study of rotating electric
//! machinery from the combined viewpoints of thermal and
//! electrica l processes from the last middle century [51, 52]. The
//! model used in this paper is proposed by [32], the electrical
//! equation can write as:
//! V a = R a 0 (1 + cu  )i a + la
//!
//! di a
//! dt
//!
//! + ke
//!
//! (1)
//!
//! Where: Va is armature voltage, R a0 is armature resistance
//! at ambient temperature, \alpha cu (\alphacu = 0.004 /°C) temperature
//! coefficient of resistance,  temperature above ambient, ia
//!
//! armature current, la is armature inductance, ke is torque
//! constant, and  armature speed. The mechanical equation:
//!
//! J
//!
//! d
//! + b + TL = ke ia
//! dt
//!
//! (2)
//!
//! where J (kg x m 2 ) is total inertia, b (N x m x s) is the viscous
//! friction constant, and TL (N x m) is the load torque.
//! The thermal model is derived by considering the power
//! dissipation and heat transfer [32]. The power dissipated by the
//! armature current flowing through the armature resistance,
//! which varies in proportion to the temperature. The iron loss is
//! proportional to speed squared for constant excitation
//! multiplied by the iron loss constant kir (kir = 0.0041
//! W/(rad/s)2 ). The power losses Plo include contributions from
//! copper losses and iron losses which frequency dependent:
//! 2
//!
//! Plo = Ra 0 (1 +  cu  )ia + kir 
//!
//! (3)
//!
//! 2
//!
//! Heat flow from the DC motor is either directly to the cooling
//! air and depends on the thermal transfer coefficients at zero
//! speed K (K =4.33 W/°C) and with speed KS (KS = 0 .0028
//! s/rad); The thermal power flow from the DC motor surface that
//! is proportional to the difference temperature between the
//! motor and the ambient air temperature, and the temperature
//! variation in the armature which depends on the thermal
//! capacity H (H=18 KJ/°C):
//! Plo = K (1 + KS  ) + H
//!
//! d
//!
//! (4)
//!
//! dt
//!
//! By arranging the previous eqs, we can write the equations
//! system as:
//! dia
//!
//! =-
//!
//! Ra 0 (1 +  cu )
//!
//! dt
//! d
//!
//! la
//! =
//!
//! dt
//! d
//! dt
//!
//! ia -
//!
//! ke
//! J
//!
//! =
//!
//! ia -
//!
//! b
//!
//! -
//!
//! J
//!
//! 1
//! J
//!
//! ke
//!
//! +
//!
//! la
//!
//! 1
//! la
//!
//! Va
//!
//! TL
//!
//! (5)
//!
//! Ra 0 (1 +  cu ) 2
//!
//! k
//!
//! K (1 + KS  )
//!
//! H
//!
//! H
//!
//! H
//!
//! ia + ir  2 -
//!
//! 
//!
//! 3. ANN ESTIMATOR
//! In recent years, several authors use a cascade forward
//! backpropagation neural network (CFNN) and has become very
//! popular [53-75] a CFNN proved their capability in several
//! applications and it they become their preferred choice [74].
//! Many authors [70-79], assert that the CFNN are similar to
//! FFNN, but include a weight connection from the input to each
//! layer and from each layer to the successive layers . As
//! example, a four-layer network has connections from layer 1 to
//! layer 2, layer 2 to layer 3, layer 3 to layer 4, layer 1 to layer 3,
//! layer 1 to layer 4 and layer 2 to layer 4. In addition, the fourlayer network also has connections from the input to all layers.
//! As FFNN and CFNN can potentially learn any input-output
//! relationship, but the CFNNs with more layers might learn
//! complex relationships more quickly [74-76, 80], which makes
//! it the right choice for intended for accelerated learning in NNs
//! [75]. The results obtained by Filik et al. in [80] suggest that
//!
//!
//! --- PAGE BREAK ---
//! which cascade forward back propagation method can be more
//! effective than feed-forward back propagation method in some
//! cases. And on the other hand, the FFNN cannot solve some
//! problems [77]. the reader is referred to [74-77, 79, 80] for more
//! detailed.
//!
//! DC
//! MACHINE
//!
//! Comparison
//! between
//! model and
//! NNs outputs
//!
//! NN
//! CFNN
//!
//! Noise
//!
//! 
//! 
//! 
//! 
//! w ij(t ) = 
//! 
//! 
//! 
//! 
//!
//! NN
//! Outputs
//!
//! Figure 1. Comparison between model and NN's outputs
//! In this application, the CFNN inputs are the voltage and
//! current and the outputs are the speed and the armature
//! temperature and resistance, to test the robustness and to make
//! the CFNN's inputs similar to the output of the sensor for the
//! real-time applications, a random white Gaussian noise has
//! been added to the inputs patterns.
//! 3.1 Back-propagation training algorithms
//! The backpropagation algorithm is used to form the neural
//! network such that on all training patterns, the sum squared
//! error 'E' between the actual network outputs, 'y' and the
//! corresponding desired outputs, yd, is minimized to a supposed
//! value:
//!
//! E =  ( y d - y )2
//!
//! (6)
//!
//! To get the optimal network architecture, for each layer the
//! transfer function types must be determined by trial and error
//! method. On the input and hidden layer, a hyperbolic tangent
//! sigmoid transfer function has been used, defined as:
//!
//! f (net j ) =
//!
//! 2
//! 1+ e
//!
//! -2 net j
//!
//! -1
//!
//! (7)
//!
//! where net is the weighted sum of the input unit, and f(net)
//! is the output units. For the output layer has 3 units with a pure
//! linear transfer function.
//!
//! f ( net j ) = net j
//!
//! w ij(t +1) = w ij(t ) + w ij(t )
//!
//! (9)
//!
//! The size of the weight change is exclusively determined by a
//! weight-specific, so-called 'update-value' performed as
//! follows:
//!
//! Model
//! Output
//!
//! Input
//! (Va)
//!
//! Learning rule [81, 83], for more details the reader is referred
//! to [48, 49, 81-84, 89].
//! In each iteration, the new weights are given by:
//!
//! (8)
//!
//! 3.2 Principe and rule
//! Resilient backpropagation often abridged by Rprop [49, 49,
//! 81-84] or RBP [46, 48, 58, 78, 85] was created by
//! M.Riedmiller et al in 1992 [81], is a learning heuristic [49] and
//! is a batch update algorithm [86] for supervised learning [84,
//! 87] and Rprop is a first-order optimization algorithm [88].
//! Rprop performs a local adaptation of the weight-updates
//! based on the sign of the partial derivative \partialE/\partialwij to eliminate
//! the harmful influence of the size of the partial derivative on
//! the weight step. It is based on the so-called Manhattan
//!
//! - ,
//!
//! if
//!
//! + ij(t ) ,
//!
//! if
//!
//! (t )
//! ij
//!
//! 0,
//!
//! E (t )
//! w ij
//! E (t )
//! w ij
//!
//! 0
//! 0
//!
//! (10)
//!
//! otherwise
//!
//! The second step of Rprop learning is to determine the new
//! update-values, the step size update rules are:
//!
//!  + (t -1)
//! E (t -1) E (t )
//! 
//! 
//! 
//! ,
//! if
//! 
//! 0
//! ij
//! 
//! w ij w ij
//! 
//! 
//! E (t -1) E (t )
//!  ij(t ) =  -   ij(t -1) , if
//! 
//! 0
//! w ij w ij
//! 
//! 
//!  ij(t -1) , otherwise
//! 
//! 
//!
//! (12)
//!
//! With 0 < η - < 1 < η + , For each weight, if there was a sign
//! change of the partial derivative of the total error function for
//! two successive iteration, the update value for that weight is
//! multiplied by a factor \eta-, where \eta- < 1, the preferred value of
//! the decrease factor which gives us the best results is -=0.5
//! [84, 87], but if two successive iteration produced the same
//! sign, the update value is multiplied by a factor of \eta+, where \eta+
//! > 1, the preferred value of the increase factor which gives us
//! the best results is \eta+ =1.2 [84, 87], the maximum weight step
//! is fixed to max =50, and the minimum step-size is min =10-6
//! [84, 87], for more detailed the interested reader is referred to
//! [49, 81-84, 87, 89].
//! 3.3 Rprop Variants
//! Two variants have been firstly created, with weightbacktracking [83, 84] named Rprop + [82] and without weightbacktracking [87] named Rprop - [82]. A performance
//! comparative studies between these algorithms and many other
//! of feedforward supervised learning techniques for many
//! benchmark problems has been presented in [84, 87].
//! Igel et al create two new versions is based to adding a stored
//! the previous error E(t-1) as a new variable to Rprop+, this
//! version named iRprop+ [82] , the second one is that the
//! derivative (\partialE(t)/\partialwij ) is set to zero [82], iRPROP- is described
//! [49, 82], so, the only difference between Rprop- and iRprop-
//! is that the derivative (\partialE (t)/\partialwij ) is set to zero [82], and as
//! comparison between iRprop- and iRprop+, iRprop- is the
//! same as iRprop+, but without weight-backtracking [49]. The
//!
//!
//! --- PAGE BREAK ---
//! reader is referred to [49,81-87] too well understood these
//! variants, where a performances comparison of all Rprop
//!
//! variants and several learning algorithms has been ca rried out
//! with four neural network benchmark problems.
//!
//! Initialisation ( - = 0.5 , \eta+= 1,2,
//! max =50 , min =10 -6 ,...)
//!
//! ∂E
//!
//! ∂ωij
//!
//! if
//!
//! ∂E
//!
//! ∂s
//!
//! ∂ net
//!
//! = ∂ s . ∂neti ∂ ω i
//! i
//!
//! i
//!
//! ∂Eሺt -1ሻ ∂E ሺtሻ
//!
//! .
//!
//! ij
//!
//! >0
//!
//! ∂ ωij
//! ∂ωij
//! ሺ tሻ
//! ሺt-1ሻ
//! Δij = minሺη + . Δij , Δ⬚
//! max ሻ
//! ሺ tሻ
//! ሺ tሻ
//! ∂E ሺtሻ
//! Δwij = -sign
//! *Δij
//! ∂ ωij
//! ሺt+1ሻ
//! ሺ tሻ
//! ሺ tሻ
//! wij
//! = wij + Δwij
//! ∂Eሺt -1ሻ ∂Eሺtሻ
//!
//! else if
//!
//! .
//!
//! ∂ ωij
//!
//! <0
//!
//! Compute the error partial derivative
//! E with respect to each weight ωij as
//! follows:
//! ∂E
//! ∂E ∂ s ∂neti
//! = . i
//! ∂ωij
//!
//! ∂si ∂neti ∂ωij
//!
//! Wher :
//! si is the output.
//! Net is the weighted sum of the inputs
//! of neuron i.
//!
//! ∂ωij
//! ሺ tሻ
//! ሺt-1ሻ
//! Δ ij = maxሺη - . Δij , Δ⬚
//! min ሻ
//! ሺt+1ሻ
//! ሺ tሻ
//! ሺt-1ሻ
//! wij
//! = wij - Δwij
//! ሺ tሻ
//!
//! ∂E
//!
//! =0
//!
//! ∂ ωij
//!
//! else if
//!
//! ∂Eሺt -1ሻ
//! ∂ ωij
//!
//! ሺ tሻ
//!
//! Δwij = -sign
//! ሺt+1ሻ
//!
//! wij
//!
//! .
//!
//! ∂Eሺtሻ
//! ∂ωij
//!
//! ∂E ሺtሻ
//!
//! ∂ ωij
//!
//! ሺ tሻ
//!
//! =0
//! ሺ tሻ
//!
//! *Δij
//! ሺ tሻ
//!
//! = wij + Δwij
//!
//! No
//!
//! Repeter les etapes pour chaque
//! iteration jusqu'a la convergence
//! ou un critere d'arret est verifie
//! (nombre d'iteration maximal,
//! erreur minimale, ..).
//!
//! Yes
//!
//! End of learning
//!
//! Figure 2. Procedures and steps of ANN based on a Resilient backpropagation learning algorithm
//!
//! 4. SIMULATION RESULTS
//! The procedure how the simulation data were used to train
//! the NN is the cross-validation error checked for multiple sets
//! of training data, this data is the result of the equation (4) with
//! the use of the parameters of BDC motor shown in table 1.
//! The estimated speed, armature temperature and resistance
//! are shown in Figs. 3-6 for a continuous running duty or
//! abbreviated by duty type S1. where duty type S1 characterized
//!
//! by an operation at a constant load maintained for sufficient
//! time to allow the machine to reach thermal equilibrium [90].
//! Table 1. Parameters of BDC motor used in the simulation.
//! Rated voltage
//! Power
//! Rated torque
//! Armature resistance
//! Armature inductance
//!
//! Va = 240 V
//! P =3 kW
//! TL = 11 N.m
//! Ra = 3.5 \Omega
//! la = 34 mH
//!
//!
//! --- PAGE BREAK ---
//! The estimated speed and the corresponding errors are shown
//! in Figure 3, the results obtained by Acarnley et al. in [32]
//! suggest that the speed estimation error from EKF is
//! approximately 2%. P. P. Acarnley assert that this application
//! is limited when a low accurate is needed such as some generalpurpose applications, not suitable for high-performance servo
//! drives [32]. However, in our results, the error is less than 0.015
//! rad/s and represent only 0.0067% of the final value as it is
//! depicting by Figure 6.
//! Speed estimation
//!
//! Figure 5 depicts the estimate resistance by NN and the
//! model response, from this figure, it can be seen that the
//! resistance has the same curvature as the armature temperature,
//! wherein the steady state the estimated resistance reached
//! almost 4,56 \Omega less than 0.04 \Omega of simulated resistance,
//! practically, this difference is negligible quantity and represents
//! only 0.9 % of the final value, this results in this paper are more
//! precise than the Zhang et al. results presented in [30], also this
//! results are in agreement with the Karanayil et al. results
//! presented in [91], where the errors of estimation of the rotor
//! and sta tor resistances is 0.3% and 5% respectively.
//!
//! 300
//! Resistance estimation
//! 5
//!
//! Real
//! Rp
//!
//! 100
//! 0
//! -100
//!
//! 0
//!
//! 20
//!
//! 40
//!
//! 60
//!
//! 80
//!
//! 100
//!
//! 120
//!
//! 140
//!
//! 160
//!
//! 180
//!
//! 200
//!
//! Real
//! Rp
//!
//! resistance, homes
//!
//! speed, rad/s
//!
//! 200
//!
//! 4.5
//!
//! 4
//!
//! time, min
//! 221.18
//!
//! 150
//! 100
//! 50
//! 0
//!
//! 0.1
//!
//! 0.2
//!
//! 0.3
//!
//! 0.4
//!
//! 20
//!
//! Real
//! Rp
//!
//! 221.16
//!
//! 40
//!
//! 100
//! time, min
//!
//! 221.12
//! 221.1
//! 190
//!
//! 140
//!
//! 195
//!
//! 200
//!
//! Real
//! Rp
//!
//! 4
//!
//! 3.5
//!
//! 3
//!
//! 0
//!
//! 50
//!
//! 100
//!
//! 4.6
//! 4.55
//! 4.5
//!
//! 150
//!
//! Real
//! Rp
//!
//! 120
//!
//! 140
//!
//! errors
//!
//! 0
//! speed
//! temperature
//! resistance
//!
//! -100
//!
//! -300
//!
//! 0
//!
//! 20
//!
//! 40
//!
//! 60
//!
//! 80
//!
//! 100
//! 120
//! time, min
//!
//! steady state estimation errors
//!
//! 20
//!
//! 140
//!
//! 160
//!
//! 180
//!
//! 200
//!
//! time, min
//! Transient temperature estimation
//! steady state temperature estimation
//!
//! speed
//! temperature
//! resistance
//!
//! 2
//! errors
//!
//! 120
//!
//! 1.5
//! 1
//!
//! 60
//! Real
//! Rp
//!
//! 40
//! 20
//! 0
//! 0
//!
//! 50
//!
//! time, min
//!
//! 100
//!
//! temperature, Deg C
//!
//! 80
//!
//! 80
//!
//! 0.5
//!
//! Real
//! Rp
//!
//! 79
//!
//! percentage error %
//!
//! 2.5
//!
//! 100
//!
//! 0
//! 150
//!
//! 160
//!
//! 170 180
//! time, min
//!
//! 190
//!
//! 140
//!
//! 160
//!
//! 180
//!
//! 200
//!
//! steady state estimation percentage errors
//! 0.04
//!
//! 3
//!
//! 80
//!
//! 200
//!
//! Speed,temperature and resistance estimation errors
//!
//! 40
//!
//! 60
//!
//! 180
//!
//! 100
//!
//! -200
//!
//! 40
//!
//! 160
//! time, min
//!
//! Figure 6 shows the estimation errors of speed, temperature
//! and resistance, and their percentage in relation to their nominal
//! value, this figure shows more clearly the perfect agreement
//! between the model outputs and the intelligent sensor outputs.
//!
//! Real
//! Rp
//!
//! 60
//!
//! 200
//!
//! Figure 5. Estimated and simulated armature resistance
//!
//! 80
//!
//! 20
//!
//! 180
//!
//! 4.65
//!
//! time, min
//!
//! Temperature estimation
//!
//! 0
//!
//! 160
//!
//! steady state resistance estimation
//!
//! time, min
//!
//! Figure 4 presents the estimated armature temperature of a
//! DC machine based on NN. As shown in Figure 4, the estimated
//! temperature reaches 77 °C, and the model output nearby in the
//! vicinity of 80 °C, while the steady state estimated error is less
//! than 3 °C as can be seen from Figs. 6. However, Nestler et al.
//! in [28] use a Luenberger's observer and it was shown that the
//! estimated winding temperature error is important, and the
//! results offered by Acarnley et al. in [32] concentrated in the
//! same context and suggest that the temperature estimation error
//! from EKF is 3 °C is approximately 3.75%.
//!
//! 0
//!
//! 120
//!
//! 4.7
//!
//! Figure 3. Estimated and simulated speed
//!
//! temperature, Deg C
//!
//! 80
//!
//! 4.5
//!
//! time, min
//!
//! temperature, Deg C
//!
//! 60
//!
//! transient resistance estimation
//!
//! 221.14
//!
//! 221.08
//! 185
//!
//! 0
//!
//! 0
//!
//! resistance, homes
//!
//! Real
//! Rp
//!
//! resistance, homes
//!
//! speed, rad/s
//!
//! speed, rad/s
//!
//! 250
//! 200
//!
//! 3.5
//!
//! steady state speed estimation
//!
//! transient speed estimation
//!
//! 200
//!
//! 0.03
//!
//! speed
//! temperature
//! resistance
//!
//! 0.02
//! 0.01
//! 0
//! -0.01
//! 150
//!
//! 160
//!
//! 170 180
//! time, min
//!
//! 190
//!
//! 200
//!
//! 78
//!
//! Figure 6. Speed, temperature and resistance estimation errors.
//!
//! 77
//! 170
//!
//! 180
//!
//! 190
//!
//! 200
//!
//! time, min
//!
//! Figure 4. Estimated and simulated armature temperature.
//!
//! The following Table 2 summarizes the simulation errors in
//! the steady state for all the estimated quantities by the ANN of
//!
//!
//! --- PAGE BREAK ---
//! CFNN type based on a resilient backpropagation learning
//! algorithm.
//! Table 2. Synopsis the estimation errors in steady-state
//! Absolute error
//! 0.015 rad/s
//! 3 0C
//! 0.04 \Omega
//!
//! Speed
//! Temperature
//! Resistance
//!
//! Relative error
//! 0.0067%
//! 3.75%
//! 0.9%
//!
//! Table. Summary of the
//!
//! 5. CONCLUSIONS
//! A sensorless speed and armature winding quantity estimator
//! is proposed for brushed DC machines based on CFNN trained
//! by RBP. The proposed estimator includes a sensorless speed
//! estimation, average armature temperature and resistance
//! estimations based only on the voltage and the current
//! measurements. The estimated speed and temperature eliminate
//! the need for speed measurements and the need for the thermal
//! sensor. In addition, the estimated temperature solves the
//! problems of obtaining the thermal information from the
//! rotating armature. Furthermore, the estimated resistance can
//! be used to improve the accuracy of the control algorithms
//! which are affected by an increase in resistance as a function of
//! temperature. The good agreement between the model and the
//! intelligent estimator demonstrates the efficiency of the
//! proposed approach.
//! ACKNOWLEDGMENT
//! This work was supported in part by: Laboratory Automation
//! Systems (LAS) in the Electrical Engineering Department,
//! Ferhat Abbas Setif1 University under the grant : A01 L07
//! UN1901 2019 0002, in other part by LGEER Laboratory,
//! Hassiba Benboua li University, Chlef, Algeria, under the
//! tutelage of the Algerian ministry of research and High
//! education.
//! REFERENCES
//! [1] Welch, R.J., Younkin, G.W. (2002). How temperature
//! affects a servomotor's electrical and mechanical time
//! constants. In Conference Record of the 2002 IEEE
//! Industry Applications Conference. 37th IAS Annual
//! Meeting (Cat. No. 02CH37344). Pittsburgh, PA, USA. 2:
//! 1041-1046. http://doi.org/10.1109/IAS.2002.1042686
//! [2] Ali, S.N., Hanif, A., Ahmed, Q. (2016). Review in
//! thermal effects on the performa nce of electric motors. In
//! 2016 International Conference on Intelligent Systems
//! Engineering (ICISE). Islamabad, Pakistan, 83-88.
//! http://doi.org/10.1109/INTELSE.2016.7475166
//! [3] Stone, G.C., Culbert, I., Boulter, E.A., Dhirani, H.
//! (2014). Electrical Insulation for Rotating Machines. John
//! Wiley
//! & Sons, Inc., Hoboken, NJ, USA.
//! http://doi.org/10.1002/9781118886663
//! [4] de Abreu, J.P.G., Emanuel, A.E. (2002). Induction motor
//! thermal aging caused by voltage distortion and
//! imbalance: Loss of useful life and its estimated cost.
//!
//! IEEE Transactions on Industry Applications. 38(1): 1220. http://doi.org/10.1109/ICPS.2001.966519
//! [5] Grubic, S., Aller, J.M., Lu, B., Habetler, T.G. (2008). A
//! survey on testing and monitoring methods for sta tor
//! insulation systems of low-voltage induction machines
//! focusing on turn insulation problems. IEEE Transactions
//! on Industrial Electronics.
//! 55(12): 4127-4136.
//! http://doi.org/10.1109/TIE.2008.2004665
//! [6] Zocholl, S.E. (2006). Understanding service factor,
//! thermal models, and overloads. In 59th Annual
//! Conference for Protective Relay Engineers, 2006.
//! College
//! Station,
//! TX,
//! USA.
//! 3-pp.
//! http://doi.org/10.1109/CPRE.2006.1638698
//! [7] Valenzuela, M.A., Verbakel, P.V., Rooks, J.A. (2003).
//! Thermal evaluation for applying TEFC induction motors
//! on short-time and intermittent duty cycles. IEEE
//! Transactions on Industry Applications. 39(1): 45-52.
//! http://doi.org/10.1109/TIA.2002.807244
//! [8] Gnacinski, P. (2008). Windings temperature and loss of
//! life of an induction machine under voltage unbalance
//! combined with
//! over-or
//! undervoltages. IEEE
//! Transactions on Energy Conversion. 23(2): 363-371.
//! http://doi.org/10.1109/TEC.2008.918596
//! [9] Zhang, P., Du, Y., Dai, J., Habetler, T.G., Lu, B. (2009).
//! Impaired-cooling-condition detection using DC-signa l
//! injection for soft-starter-connected induction motors.
//! IEEE Transactions on Industrial Electronics. 56(11):
//! 4642-4650. http://doi.org/10.1109/TIE.2009.2021588
//! [10] Bonnett, A.H., Soukup, G.C. (1992). Ca use and analysis
//! of stator and rotor failures in three-phase squirrel-cage
//! induction motors. IEEE Transactions on Industry
//! Applications.
//! 28(4),
//! 921-937.
//! http://doi.org/10.1109/28.148460
//! [11] IEEE Std 119-1974 (1975). IEEE Recommended
//! Practice for General Principles of Temperature
//! Measurement as Applied to Electrical Apparatus.
//! http://doi.org/10.1109/IEEESTD.1975.81090
//! [12] Yahoui, H., Grellet, G. (1997). Measurement of physical
//! signals in the rotating part of an electrical machine by
//! means of optical fibre transmission. Measurement. 20(3):
//! 143-148. http://doi.org/10.1016/S0263-2241(97)000195
//! [13] Compton, F.A. (1943). Temperature limits and
//! measurements for rating of DC machines. Electrical
//! Engineering.
//! 62(12),
//! 780-785.
//! http://doi.org/10.1109/EE.1943.6436045
//! [14] Bucci,
//! G.,
//! Landi, C.
//! (1996). Metrological
//! characterization of a contactless smart thrust and speed
//! sensor for linear induction motor testing. IEEE
//! Transactions on Instrumentation and Measurement.
//! 45(2): 493-498. http://doi.org/10.1109/19.492774
//! [15] AIEE committee report. (1949). Temperature rise values
//! for D-C machines. Electrical Engineering. 68(1): 206218. http://doi.org/10.1109/EE.1949.6444869.
//! [16] AIEE committee report. (1949). Temperature rise values
//! for D-C machines-II. Transactions of the American
//! Institute of Electrical Engineers. 68(2): 1118-1125.
//! http://doi.org/10.1109/T-AIEE.1949.5060060
//! [17] Ganchev, M., Kral, C., Oberguggenberger, H., Wolbank,
//! T. (2011). Sensorless rotor temperature estimation of
//! permanent magnet synchronous motor. In IECON 2011 37th Annual Conference of the IEEE Industrial
//! Electronics Society. Melbourne, VIC, Australia . 20182023. http://doi.org/10.1109/IECON.2011.6119449
//!
//!
//! --- PAGE BREAK ---
//! [18] Ganchev, M., Kubicek, B., Kappeler, H. (2010). Rotor
//! temperature monitoring system. In The XIX
//! International Conference on Electrical Machines-ICEM
//! 2010.
//! Rome,
//! Italy
//! .1-5.
//! http://doi.org/10.1109/ICELMACH.2010.5608051
//! [19] Fiorucci, E., Bucci, G., Ciancetta, F., Gallo, D., Landi,
//! C., Luiso, M. (2013). Variable Speed Driv e
//! Characterization: Review of Measurement Techniques
//! and Future Trends. Advances in Power Electronics.
//! 2013. 1-14. http://doi.org/10.1155/2013/968671
//! [20] Gao, Z., Habetler, T.G., Harley, R.G., Colby, R.S.
//! (2008). A sensorless rotor temperature estimator for
//! induction machines based on current harmonic spectral
//! estimation scheme. IEEE Transactions on Industrial
//! Electronics.
//! 55(1):
//! 407-416.
//! http://doi.org/10.1109/TIE.2007.896282
//! [21] Gao, Z., Habetler, T.G., Harley, R.G., Colby, R.S.
//! (2008). A sensorless adaptive stator winding temperature
//! estimator for mains-fed induction machines with
//! continuous-operation periodic duty cycles. IEEE
//! Transactions on Industry Applications. 44(5): 15331542. http://doi.org/10.1109/TIA.2008.2002208
//! [22] Acarnley, P., Watson, J.F. (2006). Review of positionsensorless operation of brushless permanent-magnet
//! machines. IEEE Transactions on Industrial Electronics .
//! 53(2):
//! 352-362.
//! http://doi.org/10.1109/TIE.2006.870868
//! [23] Gamazo-Real, J.C., Vazquez-Sanchez, E., Gomez-Gil, J.
//! (2010). Position and speed control of brushless DC
//! motors using sensorless techniques and application
//! trends.
//! sensors,
//! 10(7):
//! 6901-6947.
//! http://doi.org/10.3390/s100706901
//! [24] Wang, Y., Zhang, X., Yuan, X., Liu, G. (2011). Positionsensorless hybrid sliding-mode control of electric
//! vehicles with brushless DC motor. IEEE Transactions on
//! Vehicular
//! Technology.
//! 60(2):
//! 421-432.
//! http://doi.org/10.1109/TVT.2010.2100415
//! [25] Knezevic, J.M. (2013). Low-cost low-resolutio n
//! sensorless positioning of dc motor drives for vehicle
//! auxiliary applications. IEEE Transactions on Vehicular
//! Technology.
//! 62(9):
//! 4328-4335.
//! http://doi.org/10.1109/TVT.2013.2268716
//! [26] Vazquez-Sanchez, E., Gomez-Gil, J., Gamazo-Real,
//! J.C., Diez-Higuera , J.F. (2012). A new method for
//! sensorless estimation of the speed and position in
//! brushed dc motors using support vector machines. IEEE
//! Transactions on Industrial Electronics. 59(3): 13971408. http://doi.org/10.1109/TIE.2011.2161651
//! [27] Obeidat, M. A., Wang, L.Y., Lin, F. (2013). Real-time
//! parameter estimation of PMDC motors using quantized
//! sensors. IEEE transactions on vehicular technology.
//! 62(7):
//! 2977-2986.
//! http://doi.org/10.1109/TVT.2013.2251431
//! [28] Nestler, H., Sattler, P.K. (1993). On-line-estimation of
//! temperatures in electrical machines by an observer.
//! Electric machines and power systems. 21(1): 39-50.
//! http://doi.org/10.1080/07313569308909633
//! [29] Pantonial, R., Kilantang, A., Buenaobra, B. (2012). Real
//! time thermal estimation of a Brushed DC Motor by a
//! steady-state Kalman filter algorithm in multi-rate
//! sampling scheme. In TENCON 2012 IEEE Region 10
//! Conference.
//! Cebu,
//! Philippines.
//! 1-6.
//! http://doi.org/10.1109/TENCON.2012.6412194
//!
//! [30] Zhang, W., Gadsden, S.A., Habibi, S.R. (2013).
//! Nonlinear estimation of stator winding resistance in a
//! brushless DC motor. In 2013 American Control
//! Conference. Washington, DC, USA. 4699-4704.
//! http://doi.org/10.1109/ACC.2013.6580564
//! [31] French, C., Acarnley, P. (1996). Control of permanent
//! magnet motor drives using a new position estimation
//! technique. IEEE Transactions on Industry Applications.
//! 32(5): 1089-1097. http://doi.org/10.1109/28.536870
//! [32] Acarnley, P.P., Al-Tayie, J.K. (1997). Estimation of
//! speed and armature temperature in a brushed DC driv e
//! using the extended Kalman filter. IEE Proceedings Electric
//! Power
//! Applications.
//! 144(1): 13-20.
//! http://doi.org/10.1049/ip-epa:19970927
//! [33] Julier, S.J., Uhlmann, J.K. (1997). New extension of the
//! Kalman filter to nonlinear systems. In Signal processing,
//! sensor fusion, and target recognition VI International.
//! SPIE. Orlando, Florida, USA. 3068: 182-193.
//! http://doi.org/10.1117/12.280797
//! [34] Bolognani, S., Tubiana, L., Zigliotto, M. (2003).
//! Extended kalman filter tuning in sensorless PMSM
//! drives. IEEE Transactions on Industry Applications.
//! 39(6):
//! 1741-1747.
//! http://doi.org/10.1109/TIA.2003.818991
//! [35] Haseltine, E.L., Rawlings, J.B. (2005). Critical
//! evaluation of extended Kalman filtering and movinghorizon estimation. Industrial & engineering chemistry
//! research.
//! 44(8):
//! 2451-2460.
//! http://doi.org/10.1021/ie034308l
//! [36] Peroutka, Z., Smidl, V., Vosmik, D. (2009). Challenges
//! and limits of extended Kalman Filter based sensorless
//! control of permanent magnet synchronous machine
//! drives. In 2009 13th European Conference on Power
//! Electronics and Applications. Barcelona, Spain. 1-11.
//! http://doi.org/10.1109/ACC.2013.6580564
//! [37] Hendeby, G., Gustafsson, F. (2005). Fundamental
//! filtering
//! limitations
//! in
//! linear
//! non-Gaussian
//! systems. IFAC
//! Proceedings,
//! 38(1):
//! 273-278.
//! http://doi.org/10.3182/20050703-6-CZ-1902.00046
//! [38] Mellah, H., Hemsas, K.E., Taleb, R. (2016). Intelligent
//! sensor based Bayesian neural network for combined
//! parameters and states estimation of a brushed dc motor.
//! International Journal of Advanced Computer Science and
//! Applications(IJACSA),
//! 7(7):230-235.
//! http://doi.org/10.14569/IJACSA.2016.070731
//! [39] Bouchiba, B., Bousserhane, I.K., Fellah, M.K., Hazzab,
//! A. (2017). Artificial neural network sliding mode control
//! for multi-machine web winding system. Revue
//! Roumaine
//! des
//! Sciences
//! Techniques-Serie
//! Electrotechnique et Energetique, 62(1): 109-113.
//! [40] Vas, P. (1999). Artificial-intelligence-based electrical
//! machines and drives: application of fuzzy, neural, fuzzy neural, and genetic-algorithm-based technique, Oxford
//! University Press.
//! [41] Azzeddine, H.A., Tioursi, M., Chaouch, D.E., Khiari, B.
//! (2016). An offline trained artificial neural network to
//! predict a photovoltaic panel maximum power point. Rev.
//! Roum. Sci. Techn.-Electrotechn. et Energ, 61(3): 255257
//! [42] Bose, B.K. (1994). Expert system, fuzzy logic, and
//! neural network applications in power electronics and
//! motion control. Proceedings of the IEEE, 82(8): 13031323. http://doi.org/10.1109/5.301690
//!
//!
//! --- PAGE BREAK ---
//! [43] Florea, B.F., Grigore, O., Datcu, M. (2017). Learning
//! online spatial exploration by optimizing artificial neural
//! networks assisted by a pheromone map. revue roumaine
//! des sciences techniques-serie electrotechnique et
//! energetique, 62(2): 209-214
//! [44] Kayri, M. (2016). Predictive abilities of bayesian
//! regularization and Levenberg-Marquardt algorithms in
//! artificial neural networks: a comparative empirical study
//! on social data. Mathematical and Computational
//! Applications,
//! 21(2):
//! 1-11.
//! http://doi.org/10.3390/mca21020020
//! [45] Afram, A., Janabi-Sharifi, F., Fung, A.S., Raahemifar, K.
//! (2017). Artificial neural network (ANN) based model
//! predictive control (MPC) and optimization of HVAC
//! systems: A state of the art review and case study of a
//! residential HVAC system. Energy and Buildings, 141,
//! 96-113. http://doi.org/10.1016/j.enbuild.2017.02.012
//! [46] Wang, S.H., Du, S., Zhang, Y., Phillips, P., Wu, L.N.,
//! Chen, X.Q., Zhang, Y.D. (2017). Alzheimer's disease
//! detection by pseudo Zernike moment and linear
//! regression
//! classification. CNS & Neurologica l
//! Disorders-Drug Targets (Formerly Current Dru g
//! Targets-CNS & Neurological Disorders), 16(1), 11-15.
//! http://doi.org/10.2174/1871527315666161111123024
//! [47] Barati-Harooni,
//! A.,
//! Najafi-Marghmaleki,
//! A.,
//! Mohammadi, A.H. (2017). Prediction of heat capacities
//! of ionic liquids using chemical structure based networks.
//! Journal of Molecular Liquids, 227: 324-332.
//! http://doi.org/10.1016/j.molliq.2016.11.119
//! [48] Patnaik, L.M., Rajan, K. (2000). Target detection
//! through image processing and resilient propagation
//! algorithms. Neurocomputing, 35(1-4): 123-135.
//! http://doi.org/10.1016/S0925-2312(00)00301-5
//! [49] Igel, C., Husken, M. (2003). Empirical evaluation of the
//! improved Rprop learning algorithms. Neurocomputing,
//! 50:105-123.
//! http://doi.org/10.1016/S09252312(01)00700-7
//! [50] Liu, Q., Liu, G., Li, L., Yuan, X.T., Wang, M., Liu, W.
//! (2017). Reversed spectral hashing. IEEE transactions on
//! neural networks and learning systems, 29(6): 2441-2449.
//! http://doi.org/10.1109/TNNLS.2017.2696053
//! [51] Kaye, J., Gouse, S.W. (1956). Thermal Analysis of a
//! Small DC Motor; Part I. Dimensional Analysis of
//! Combined Thermal and Electrical Processes [includes
//! discussion]. Transactions of the American Institute of
//! Electrical Engineers. Part III: Power Apparatus and
//! Systems,
//! 75(3):
//! 1463-1467.
//! http://doi.org/10.1109/AIEEPAS.1956.4499460.
//! [52] Kaye, J., Gouse, S.W., Elgar, E.C. (1956). Thermal
//! Analysis of a Small DC Motor; Part II. Experimental
//! Study of Steady-State Temperature Distribution in a DC
//! Motor with Correlations Based on Dimensional Analysis
//! [includes discussion]. Transactions of the American
//! Institute of Electrical Engineers. Part III: Power
//! Apparatus and
//! Systems,
//! 75(3):
//! 1468-1486.
//! http://doi.org/10.1109/AIEEPAS.1956.4499461
//! [53] Li, W., Wu, X., Jiao, W., Qi, G., Liu, Y. (2017).
//! Modelling of dust removal in rotating packed bed using
//! artificial neural networks (ANN). Applied Thermal
//! Engineering,
//! 112:
//! 208-213.
//! http://doi.org/10.1016/j.applthermaleng.2016.09.159
//! [54] Nabipour, M., Keshavarz, P. (2017). Modeling surface
//! tension of pure refrigerants using feed-forward backpropagation neural networks. International Journal of
//!
//! Refrigeration,
//! 75:
//! 217-227.
//! http://doi.org/10.1016/j.ijrefrig.2016.12.011
//! [55] Venkadesan, A., Himavathi, S., Sedhuraman, K.,
//! Muthuramalingam, A. (2017). Design and field
//! programmable gate array implementation of cascade
//! neural network based flux estimator for speed estimation
//! in induction motor drives. IET Electric Power
//! Applications, 11(1): 121-131. http://doi.org/10.1049/ietepa.2016.0550
//! [56] Sundaram, N.M., Sivanandam, S.N., Renupriya, V.
//! (2016). Artificial neural network approach for dynamic
//! modelling of heat exchanger for data prediction. Indian
//! Journal of Science and Technology, 9(S1): 1-7.
//! http://doi.org/10.17485/ijst/2016/v9iS1/86189
//! [57] Sun, C., He, W., Ge, W., Chang, C. (2016). Adaptive
//! neural network control of biped robots. IEEE
//! transactions on systems, man, and cybernetics: systems,
//! 47(2):
//! 315-326.
//! http://doi.org/10.1109/TSMC.2016.2557223
//! [58] Saeedi, E., Hossain, M.S., Kong, Y. (2016). Side-channel
//! information characterisation based on cascade-forward
//! back-propagation neural network. Journal of Electronic
//! Testing, 32(3), 345-356. http://doi.org/10.1007/s10836016-5590-4
//! [59] Hamzic, A., Avdagic, Z. (2016). Multilevel prediction of
//! missing time series dam displacements data based on
//! artificial neural networks voting evaluation. In 2016
//! IEEE International Conference on Systems, Man, and
//! Cybernetics (SMC). Budapest, Hungary. 002391002396.
//! IEEE.
//! http://doi.org/10.1109/SMC.2016.7844597
//! [60] Hussain, W., Hussain, F., Hussain, O. (2016). QoS
//! prediction methods to avoid SLA violation in post interaction time phase. In 2016 IEEE 11th Conference on
//! Industrial Electronics and Applications (ICIEA). Hefei,
//! China .
//! 32-37.
//! IEEE.
//! http://doi.org/10.1109/ICIEA.2016.7603547
//! [61] Agarwal, A., Sharma, A.K., Khandelwal, S. (2016).
//! Fingerprint recognition system by termination points
//! using cascade-forward backpropagation neural network.
//! In Proceedings of the International Congress on
//! Information and Communication Technology. 203-211.
//! Springer, Singapore.
//! [62] Shelke, S., Apte, S. (2016). Performance optimization
//! and comparative analysis of neural networks for
//! handwritten Devanagari character recognition. In 2016
//! International Conference on Signal and Information
//! Processing (IConSIP). Vishnupuri, India, 1-5. IEEE.
//! [63] Pertl, M., Heussen, K., Gehrke, O., Rezkalla, M. (2016,
//! July). Voltage estimation in active distribution grid s
//! using neural networks. In 2016 IEEE Power and Energy
//! Society General Meeting (PESGM) . Boston, MA,
//! USA.1-5. IEEE.
//! [64] Narad, S., Chavan, P. (2016). Cascade forward back propagation neural network based group authentication
//! using (n, n) secret sharing scheme. Procedia Computer
//! Science, 78: 185-191.
//! [65] Shoumy, N.J., Yaakob, S.N., Ehkan, P., Ali, M.S.,
//! Khatun, S. (2016). Cascade-forward neural network
//! performance study for bloodstain image analysis. In 2016
//! 3rd International Conference on Electronic Design
//! (ICED). Phuket, Thailand. 245-250. IEEE.
//! [66] Taghavifar, H., Mardani, A., Taghavifar, L. (2013). A
//! hybridized artificial neural network and imperialist
//!
//!
//! --- PAGE BREAK ---
//! competitive algorithm optimization approach for
//! prediction of soil compaction in soil bin facility.
//! Measurement, 46(8): 2288-2299.
//! [67] Chayjan, R.A., Esna -Ashari, M. (2010). Modeling
//! isosteric heat of soya bean for desorption energy
//! estimation using neural network approach. Chilean
//! journal of agricultural research, 70(4): 616-625.
//! [68] Aziz, M.A., Ismail, N., Yassin, I.M., Zabidi, A., Ali,
//! M.M. (2015). Agarwood oil quality classification using
//! cascade-forward neural network. In 2015 IEEE 6th
//! Control and System Graduate Research Colloquium
//! (ICSGRC). Shah Alam, Malaysia . 112-115.
//! [69] Saini, S., Vijay, R. (2015). Mammogram analysis using
//! feed-forward back propagation and cascade-forward
//! back propagation artificial neural network. In 2015 fifth
//! international conference on communication systems and
//! network technologies. Gwalior, India . 1177-1180. IEEE.
//! http://doi.org/10.1109/CSNT.2015.78
//! [70] Sciuto, G.L., Cammarata, G., Capizzi, G., Coco, S.,
//! Petrone, G. (2016). Design optimization of solar chimney
//! power plant by finite elements based numerical model
//! and cascade neural networks. In 2016 International
//! Symposium on Power Electronics, Electrical Drives,
//! Automation and Motion (SPEEDAM). Anacapri, Italy.
//! 1016-1022.
//! IEEE.
//! http://doi.org/10.1109/SPEEDAM.2016.7526002
//! [71] Singh, S., Vishwakarma, D.N. (2016). ANN and wavelet
//! entropy based approach for fault location in series
//! compensated lines. In 2016 International Conference on
//! Microelectronics, Computing and Communications
//! (MicroCom).
//! Durgapur,
//! India .
//! 1-6.
//! IEEE.
//! http://doi.org/10.1109/MicroCom.2016.7522557
//! [72] Capizzi, G., Sciuto, G.L., Monforte, P., Napoli, C.
//! (2015). Cascade feed forward neural network-based
//! model for air pollutants evaluation of single monitoring
//! stations in urban areas. International Journal of
//! Electronics and Telecommunications. 61(4): 327-332.
//! http://doi.org/10.1515/eletel-2015-0042
//! [73] Lashkarbolooki, M., Shafipour, Z.S., Hezave, A.Z.
//! (2013). Trainable cascade-forward back-propagation
//! network modeling of spearmint oil extraction in a packed
//! bed using SC-CO2. The Journal of Supercritical Fluids .
//! 73:
//! 108-115.
//! http://doi.org/10.1016/j.supflu.2012.10.013
//! [74] Pwasong, A., Sathasivam, S. (2016). A new hybrid
//! quadratic
//! regression
//! and
//! cascade
//! forward
//! backpropagation neural network. Neurocomputing, 182:
//! 197-209. http://doi.org/10.1016/j.neucom.2015.12.034
//! [75] Khaki, M., Yusoff, I., Islami, N., Hussin, N.H. (2016).
//! Artificial neural network technique for modeling of
//! groundwater level in Langat Basin, Malaysia. Sains
//! Malays. 45(1):19-28.
//! [76] Al-allaf, O.N.A. (2012). Cascade-forward vs. function
//! fitting neural network for improving image quality and
//! learning time in image compression system. In
//! Proceedings of the world congress on engineerin g .
//! London, U.K. 2: 4-6.
//! [77] Wilamowski, B.M. (2011). How to not get frustrated
//! with neural networks. In 2011 IEEE International
//! Conference on Industrial Technology. Auburn, AL,
//! USA. 5-11. http://doi.org/10.1109/ICIT.2011.5754336
//! [78] Yao-ming, Z., Zhi-jun, M., Xu-zhi, C., Zhe, W. (2012).
//! Helicopter engine performance prediction based on
//! cascade-forward process neural network. In 2012 IEEE
//!
//! Conference on Prognostics and Health Management.
//! Denver,
//! CO,
//! USA.
//! 1-5.
//! IEEE.
//! http://doi.org/10.1109/ICPHM.2012.6299515
//! [79] Beale, M.H., Hagan, M.T., Demuth, H.B. (2008). Neural
//! network toolbox. User's Guide, MathWorks.
//! [80] Filik, U. B., Kurban, M. (2007). A new approach for the
//! short-term load forecasting with autoregressive and
//! artificial neural network models. International Journal of
//! Computational Intelligence Research. 3(1): 66-71.
//! http://doi.org/10.5019/j.ijcir.2007.88
//! [81] Riedmiller, M., Braun, H. (1992). RPROP - A Fast
//! Adaptive Learning Algorithm. In International
//! Symposium on Computer and Information Science VII
//! (ISCIS VII). Antalya, Turkey. 279-286.
//! [82] Igel, C., Husken, M. (2000). Improving the Rprop
//! learning algorithm. In Proceedings of the second
//! international ICSC symposium on neural computation
//! (NC 2000). ICSC Academic Press. Berlin, Germany.
//! 115-121.
//! [83] Riedmiller,
//! M. (1994). Rprop-description and
//! implementation details, report, 5-6.
//! [84] Riedmiller, M., Braun, H. (1993). A direct adaptive
//! method for faster backpropagation learning: The RPROP
//! algorithm. In IEEE international conference on neural
//! networks. San Francisco, CA, USA. 586-591. IEEE.
//! http://doi.org/10.1109/ICNN.1993.298623
//! [85] Dongardive, J., Abraham, S. (2017). Reaching optimized
//! parameter set: protein secondary structure prediction
//! using neural network. Neural Computing and
//! Applications.
//! 28(8):1947-1974.
//! http://doi.org/10.1007/s00521-015-2150-2
//! [86] Anastasiadis, A.D., Magoulas, G.D., Vrahatis, M.N.
//! (2005). Sign-based learning schemes for pattern
//! classification. Pattern Recognition Letters. 26(12): 1926 1936. http://doi.org/10.1016/j.patrec.2005.03.013
//! [87] Riedmiller, M. (1994). Advanced supervised learning in
//! multi-layer perceptrons--from backpropagation to
//! adaptive learning algorithms. Computer Standards &
//! Interfaces. 16(3): 265-278. http://doi.org/10.1016/09205489(94)90017-5
//! [88] Battiti, R. (1992). First-and second-order methods for
//! learning: between steepest descent and Newton's method.
//! Neural
//! computation.
//! 4(2):
//! 141-166.
//! http://doi.org/10.1162/neco.1992.4.2.141
//! [89] Y. Hifny. ( 2013). Deep Learning Based on Manhattan
//! Update Rule, Proceedings of the 30th International
//! Conference on Machine Learning, Atlanta, Georgia,
//! USA,.
//! [90] IEC 60034 ‐ 1. (2004). Rotating Electrical Machines
//! Part 1: Rating and Performance.
//! [91] Karanayil, B., Rahman, M.F., Grantham, C. (2007).
//! Online stator and rotor resistance estimation scheme
//! using artificial neural networks for vector controlled
//! speed sensorless induction motor drive. IEEE
//! transactions on Industrial Electronics. 54(1): 167 -176.
//! http://doi.org/10.1109/TIE.2006.888778
//! [92] Magrini, A., Lazzari, S., Marenco, L., Guazzi, G. (2017).
//! A procedure to evaluate the most suitable integrated
//! solutions for increasing energy performance of the
//! building's envelope, avoiding moisture problems.
//! International Journal of Heat and Technology, 35(4):
//! 689-699. https://doi.org/10.18280/ijht.350401
//!
//!
//! --- PAGE BREAK ---
//! NOMENCLATURE
//! b
//! E
//! H
//! i
//! J
//! K
//! k
//! KS
//! ke
//! l
//! net
//! P
//! R
//! T
//! V
//! y
//!
//! viscous friction constant, N. m. s
//! sum squared error
//! thermal capacity, kJ. K-1
//! current,
//! total inertia, kg.m 2
//! thermal transfer coefficients, W. K-1
//! loss constant, W. rad -2 . s2
//! thermal transfer coefficients with speed, s.
//! rad -1
//! torque constant, V. rad -1 . s1
//! Inductance, H
//! weighted sum of the input unit
//! power, W
//! resistance, \Omega
//! torque, N. m
//! voltage, V
//! network outputs
//!
//! Greek symbols
//! \alpha
//! 
//! 
//! 
//! \eta
//!
//! temperature coefficient of resistance, K-1
//! temperature above ambient, K
//! armature speed, rad. s-1
//! weight step
//! factor
//!
//! Subscripts
//! a
//! a0
//! cu
//! d
//! ir
//! lo
//! s
//! max
//! min
//! 0
//! -
//! +
//! l
//!
//! armature
//! ambient temperature
//! Copper
//! desired
//! iron
//! losses
//! speed
//! maximum
//! minimum
//! Zero speed
//! decrease
//! increase
//! load
//!
//!
//! --- PAGE BREAK ---
//!
//! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//!
