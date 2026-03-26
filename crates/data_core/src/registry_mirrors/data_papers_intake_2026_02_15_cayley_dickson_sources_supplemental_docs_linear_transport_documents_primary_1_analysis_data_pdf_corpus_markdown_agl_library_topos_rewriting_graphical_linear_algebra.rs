//! # Extracted text: graphical_linear_algebra.pdf
//!
//! - source_root: `/home/eirikr/Documents/AGL_Library/Topos_Rewriting_Documentation`
//! - source_relpath: `graphical_linear_algebra.pdf`
//! - source_abs: `/home/eirikr/Documents/AGL_Library/Topos_Rewriting_Documentation/graphical_linear_algebra.pdf`
//! - detected_kind: `pdf`
//! - extracted_at_utc: `2026-01-02T17:31:39+00:00`
//! - pages: `30`
//! - title: `Categorical Foundations of Gradient-Based Learning`
//! - author: ``
//! - subject: `-  Software and its engineering  ->  General programming languages; -  Social and professional topics  ->  History of programming languages;`
//! - keywords: ``
//! - creation_date: `Tue Jul 13 18:16:30 2021 PDT`
//! - mod_date: `Tue Jul 13 18:16:30 2021 PDT`
//! - encrypted: `no`
//!
//! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~text
//! 1
//! Categorical Foundations of Gradient-Based Learning
//!
//! arXiv:2103.01931v2 [cs.LG] 13 Jul 2021
//!
//! G.S.H. CRUTTWELL, Mount Allison University, Canada
//! BRUNO GAVRANOVIĆ and NEIL GHANI, University of Strathclyde, UK
//! PAUL WILSON and FABIO ZANASI, University College London, UK
//! We propose a categorical semantics of gradient-based machine learning algorithms in terms of lenses,
//! parametrised maps, and reverse derivative categories. This foundation provides a powerful explanatory
//! and unifying framework: it encompasses a variety of gradient descent algorithms such as ADAM, AdaGrad,
//! and Nesterov momentum, as well as a variety of loss functions such as as MSE and Softmax cross-entropy,
//! shedding new light on their similarities and differences. Our approach to gradient-based learning has examples
//! generalising beyond the familiar continuous domains (modelled in categories of smooth maps) and can be
//! realized in the discrete setting of boolean circuits. Finally, we demonstrate the practical significance of our
//! framework with an implementation in Python.
//! Additional Key Words and Phrases: Category Theory, Semantics, Machine Learning, Lenses
//!
//! 1
//!
//! INTRODUCTION
//!
//! The last decade has witnessed a surge of interest in machine learning, fuelled by the numerous
//! successes and applications that these methodologies have found in many fields of science and
//! technology. As machine learning techniques become increasingly pervasive, algorithms and models
//! become more sophisticated, posing a significant challenge both to the software developers and the
//! users that need to interface, execute and maintain these systems. In spite of this rapidly evolving
//! picture, the formal analysis of many learning algorithms mostly takes place at a heuristic level [Seshia and Sadigh 2016], or using definitions that fail to provide a general and scalable framework for
//! describing machine learning. Indeed, it is commonly acknowledged through academia, industry,
//! policy makers and funding agencies that there is a pressing need for a unifying perspective, which
//! can make this growing body of work more systematic, rigorous, transparent and accessible both
//! for users and developers [Exp 2019; Olah 2015].
//! Consider, for example, one of the most common machine learning scenarios: supervised learning
//! with a neural network. This technique trains the model towards a certain task, e.g. the recognition
//! of patterns in a data set (cf. Figure 1). There are several different ways of implementing this
//! scenario. Typically, at their core, there is a gradient update algorithm (often called the “optimiser”),
//! depending on a given loss function, which updates in steps the parameters of the network, based
//! on some learning rate controlling the “scaling” of the update. All of these components can vary
//! independently in a supervised learning algorithm and a number of choices is available for loss
//! maps (quadratic error, Softmax cross entropy, dot product, etc.) and optimisers (Adagrad [Duchi
//! et al. 2011], Momentum [Polyak 1964], and Adam [Kingma and Ba 2015], etc.).
//! This scenario highlights several questions: is there a uniform mathematical language capturing
//! the different components of the learning process? Can we develop a unifying picture of the various
//! optimisation techniques, allowing for their comparative analysis? Moreover, it should be noted
//! that supervised learning is not limited to neural networks. For example, supervised learning
//! is surprisingly applicable to the discrete setting of boolean circuits [Wilson and Zanasi 2020]
//! where continuous functions are replaced by boolean-valued functions. Can we identify an abstract
//! Authors’ addresses: G.S.H. Cruttwell, Department of Mathematics and Computer Science, Mount Allison University, Canada;
//! Bruno Gavranović; Neil Ghani, University of Strathclyde, UK, bruno@brunogavranovic.com; Paul Wilson, paul@statusfailed.
//! com; Fabio Zanasi, University College London, UK.
//! 2018. 2475-1421/2018/1-ART1 $15.00
//! https://doi.org/
//!
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! 1:2
//!
//! G.S.H. Cruttwell, Bruno Gavranović, Neil Ghani, Paul Wilson, and Fabio Zanasi
//!
//! Fig. 1. An informal illustration of gradient-based learning. This neural network is trained to distinguish
//! different kinds of animals in the input image. Given an input 𝑋 , the network predicts an output 𝑌 , which is
//! compared by a ‘loss map’ with what would be the correct answer (‘label’). The loss map returns a real value
//! expressing the error of the prediction; this information, together with the learning rate (a weight controlling
//! how much the model should be changed in response to error) is used by an optimiser, which computes by
//! gradient-descent the update of the parameters of the network, with the aim of improving its accuracy. The
//! neural network, the loss map, the optimiser and the learning rate are all components of a supervised learning
//! system, and can vary independently of one another.
//!
//! perspective encompassing both the real-valued and the boolean case? In a nutshell, this paper seeks
//! to answer the question:
//! what are the fundamental mathematical structures underpinning gradient-based learning?
//! Our approach to this question stems from the identification of three fundamental aspects of the
//! gradient-descent learning process:
//! (I) computation is parameterised, e.g. in the most simple case we are given a function 𝑓 :
//! 𝑃 × 𝑋 → 𝑌 and learning consists of finding a parameter 𝑝 : 𝑃 such that 𝑓 (𝑝, −) is the best
//! function according to some criteria. More generally, the weights on the internal nodes of a
//! neural network are a parameter which the learning is seeking to optimize. Parameters also
//! arise elsewhere, e.g. in the loss function (see later).
//! (II) information flows bidirectionally: in the forward direction the computation turns inputs
//! via a sequence of layers into predicted outputs, and then into a loss value; in the reverse
//! direction, backpropagation is used propagate the changes backwards through the layers, and
//! then turn them into parameter updates.
//! (III) the basis of parameter update via gradient descent is differentiation e.g. in the simple case
//! we differentiate the function mapping a parameter to its associated loss to reduce that loss.
//! We model bidirectionality via lenses [Bohannon et al. 2008; Clarke et al. 2020; Hedges 2018] and
//! based upon the above three insights, we propose the notion of parametric lens as the fundamental
//! semantic structure of learning. In a nutshell, a parametric lens is a process with three kinds of
//! interfaces: inputs, outputs, and parameters. On each interface, information flows both ways, i.e.
//! computations are bidirectional. These data are best explained with our graphical representation
//! of parametric lenses, with inputs 𝐴, 𝐴 ′, outputs 𝐵, 𝐵 ′, parameters 𝑃, 𝑃 ′, and arrows indicating
//! information flow (below left). The graphical notation also makes evident that parametric lenses are
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! Categorical Foundations of Gradient-Based Learning
//!
//! 1:3
//!
//! open systems, which may be composed along their interfaces (below center and right).
//! 𝑄 𝑄′
//!
//! 𝑃 𝑃′
//! 𝑃
//!
//! 𝑄 𝑄′
//!
//! 𝑃′
//! 𝐵
//!
//! 𝐴
//! 𝐴′
//!
//! 𝐵
//! 𝐵′
//!
//! 𝐴
//! 𝐴′
//!
//! 𝐶
//! 𝐶′
//!
//! 𝐵′
//!
//! 𝑃
//! 𝐴
//! 𝐴′
//!
//! 𝑃′
//! 𝐵
//! 𝐵′
//!
//! This pictorial formalism is not just an intuitive sketch: as we will show, it can be understood as a
//! completely formal (graphical) syntax using the formalism of string diagrams [Selinger 2010], in a
//! way similar to how other computational phenomena have been recently analysed e.g. in quantum
//! theory [Coecke and Kissinger 2017], control theory [Baez and Erbele 2014; Bonchi et al. 2017], and
//! digital circuit theory [Ghica et al. 2017].
//! It is intuitively clear how parametric lens express aspects (I) and (II) above, whereas (III) will
//! be achieved by studying them in a space of ‘differentiable objects’ (in a sense that will be made
//! precise). The main technical contribution of our paper is showing how the various ingredients
//! involved in learning (the model, the optimiser, the error map and the learning rate) can be uniformly
//! understood as being built from parametric lenses.
//! 𝐴
//!
//! 𝑃
//!
//! 𝑃
//!
//! 𝐵
//!
//! Optimiser
//!
//! 𝑃′
//!
//! 𝑃
//! 𝐴
//!
//! 𝐵′
//! 𝐵
//!
//! 𝐿
//! Learning
//! Loss
//!
//! Model
//! 𝐴′
//!
//! 𝐵′
//!
//! 𝐿′
//!
//! rate
//!
//! Fig. 2. The parametric lens that captures the learning process informally sketched in Figure 1. Note each
//! component is a lens itself, whose composition yields the interactions described in Figure 1. Defining this
//! picture formally will be the subject of Sections 3-4. Also, an animation of this supervised learning system is
//! available online. 1
//!
//! We will use category theory as the formal language to develop our notion of parametric lenses,
//! and make Figure 2 mathematically precise. The categorical perspective brings several advantages,
//! which are well-known, established principles in programming language semantics [Abramsky and
//! Coecke 2004; Selinger 2001; Turi and Plotkin 1997]. Three of them are particularly important to
//! our contribution, as they constitute distinctive advantages of our semantic foundations:
//! Abstraction Our approach studies which categorical structures are sufficient to perform gradientbased learning. This analysis abstracts away from the standard case of neural networks in
//! several different ways: as we will see, it encompasses other models (namely Boolean circuits),
//! different kinds of optimisers (including Adagrad, Adam, Nesterov momentum), and error
//! maps (including quadratic and softmax cross entropy loss). These can be all understood as
//! parametric lenses, and different forms of learning result from their interaction.
//! 1 https://giphy.com/gifs/xXl4CSLvGsd7fmeTOJ
//!
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! 1:4
//!
//! G.S.H. Cruttwell, Bruno Gavranović, Neil Ghani, Paul Wilson, and Fabio Zanasi
//!
//! Uniformity As seen in Figure 1, learning involves ingredients that are seemingly quite different:
//! a model, an optimiser, a loss map, etc. We will show how all these notions may be seen
//! as instances of the categorical definition of a parametric lens, thus yielding a remarkably
//! uniform description of the learning process, and supporting our claim of parametric lenses
//! being a fundamental semantic structure of learning.
//! Compositionality The use of categorical structures to describe computation naturally enables
//! compositional reasoning whereby complex systems are analysed in terms of smaller, and
//! hence easier to understand, components. Compositionality is a fundamental tenet of programming language semantics; in the last few years, it has found application in the study of
//! diverse kinds of computational models, across different fields— see e.g. [Bonchi et al. 2017;
//! Coecke and Kissinger 2017; Ghani et al. 2016; Spivak 2010]. As made evident by Figure 2,
//! our approach models a neural network as a parametric lens, resulting from the composition
//! of simpler parametric lenses, capturing the different ingredients involved in the learning
//! process. Moreover, as all the simpler parametric lenses are themselves composable, one may
//! engineer a different learning process by simply plugging a new lens on the left or right of
//! existing ones. This means that one can glue together smaller and relatively simple networks
//! to create larger and more sophisticated neural networks.
//! We now give a synopsis of our contributions:
//! • In Section 2, we introduce the tools necessary to define our notion of parametric lens. First,
//! in Section 2.1, we introduce a notion of parametrisation for categories, which amounts to a
//! functor Para(−) turning a category C into one Para(C) of ‘parametrised C-maps’. Second,
//! we recall lenses (Section 2.2). In a nutshell, a lens is a categorical morphism equipped with
//! operations to view and update values in a certain data structure. Lenses play a prominent role
//! in functional programming [Steckermeier 2015], as well as in the foundations of database
//! theory [Johnson et al. 2012] and more recently game theory [Ghani et al. 2016]. Considering
//! lenses in C simply amounts to the application of a functorial construction Lens(−), yielding
//! Lens(C). Finally, we recall the notion of a cartesian reverse differential category (CRDC): a categorical structure axiomatising the notion of differentiation [Cockett et al. 2019] (Section 2.4).
//! We wrap up in Section 2.3, by combining these ingredients into the notion of parametric lens,
//! formally defined as a morphism in Para(Lens(C)) for a CRDC C. In terms of our desiderata
//! (I)-(III) above, note that Para(−) accounts for (I), Lens(−) accounts for (II), and the CRDC
//! structure accounts for (III).
//! • As seen in Figure 1, in the learning process there are many components at work: the model,
//! the optimiser, the loss map, the learning rate, etc.. In Section 4, we show how the previously
//! introduced notion of parametric lens provides a uniform characterisation for such components.
//! Moreover, for each of them, we show how different variations appearing in the literature
//! become instances of our abstract characterisation. The plan is as follows:
//! ◦ In Section 3.1, we show how the combinatorial model subject of the training can be
//! seen as a parametric lens. The conditions we provide are met by the ‘standard’ case of
//! neural networks, but also enables the study of learning for other classes of models. In
//! particular, another instance are Boolean circuits: learning of these structures is relevant
//! to binarisation [Courbariaux et al. [n.d.]] and it has been explored recently using a categorical approach [Wilson and Zanasi 2020], which turns out to be a particular case of our
//! framework.
//! ◦ In Section 3.2, we show how the loss maps associated with training are also parametric
//! lenses. We also show how our approach covers the cases of quadratic error, Boolean error,
//! Softmax cross entropy, but also the ‘dot product loss’ associated with the phenomenon of
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! Categorical Foundations of Gradient-Based Learning
//!
//! 1:5
//!
//! deep dreaming [Dosovitskiy and Brox 2015; Mahendran and Vedaldi 2014; Nguyen et al.
//! 2014; Simonyan et al. 2014].
//! ◦ In Section 3.3, we model the learning rate as a parametric lens. This analysis also allows
//! us to contrast how learning rate is handled in the ‘real-valued’ case of neural networks
//! with respect to the ‘Boolean-valued’ case of Boolean circuits.
//! ◦ In Section 3.4, we show how optimisers can be modelled as ‘reparametrisations’ of models
//! as parametric lenses. As case studies, in addition to basic gradient update, we consider the
//! stateful variants: Momentum [Polyak 1964], Nesterov Momentum [Sutskever et al. 2013],
//! Adagrad [Duchi et al. 2011], and Adam (Adaptive Moment Estimation) [Kingma and Ba
//! 2015]. Also, on Boolean circuits, we show how the reverse derivative ascent of [Wilson
//! and Zanasi 2020] can be also regarded in such way.
//! • In Section 4, we study how the composition of the lenses defined in Section 3 yields a
//! description of different kinds of learning processes.
//! ◦ Section 4.1 is dedicated to modelling supervised learning of parameters, in the way
//! described in Figure 1. This amounts essentially to study the composite of lenses expressed in
//! Figure 2, for different choices of the various components. In particular we show (i) quadratic
//! loss with basic gradient descent, (ii) softmax cross entropy loss with basic gradient descent,
//! (iii) quadratic loss with Nesterov momentum, and (iv) learning in Boolean circuits with
//! XOR loss and basic gradient ascent.
//! ◦ In order to showcase the flexibility of our approach, in Section 4.2 we depart from our ‘core’
//! case study of parameter learning, and turn attention to supervised learning of inputs. The
//! idea behind this technique, sometimes called deep dreaming, is that, instead of the network
//! parameters, one updates the inputs, in order to elicit a particular interpretation [Dosovitskiy
//! and Brox 2015; Mahendran and Vedaldi 2014; Nguyen et al. 2014; Simonyan et al. 2014].
//! Deep dreaming can be easily expressed within our approach, with a different rearrangement
//! of the various parametric lenses involved in the learning process, see Figure 7 below. The
//! abstract viewpoint of categorical semantics provides a mathematically precise and visually
//! captivating description of the differences between the usual parameter learning process
//! and deep dreaming.
//! • In Section 5 we describe a proof-of-concept Python implementation2 based on the theory
//! developed in this paper. This code is intended to show more concretely the payoff of our
//! approach. Model architectures, as well as the various components participating in the learning
//! process, are now expressed in a uniform, principled mathematical language, in terms of
//! lenses. As a result, computing network gradients is greatly simplified, as it amounts to lens
//! composition. Moreover, the modularity of this approach allows one to more easily tune the
//! various parameters of training.
//! We give a demonstration of our library via a number of experiments, and prove correctness
//! by achieving accuracy on par with an equivalent model in Keras, a mainstream deep learning
//! framework [Chollet et al. 2015]. In particular, we create a working non-trivial neural network
//! model for the MNIST image-classification problem [Lecun et al. 1998].
//! • Finally, in Sections 6 and 7, we discuss related and future work.
//! 2
//!
//! CATEGORICAL TOOLKIT
//!
//! In this section, we describe the three categorical components of our framework, each corresponding
//! to an aspect of gradient-based learning. In Section 2.1 we review the Para construction which builds
//! a category of parametrised maps from a monoidal category, and describe its graphical language. In
//! 2 Available at https://github.com/statusfailed/numeric-optics-python/.
//!
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! 1:6
//!
//! G.S.H. Cruttwell, Bruno Gavranović, Neil Ghani, Paul Wilson, and Fabio Zanasi
//!
//! Section 2.2, we review the Lens construction, which builds a category of “bidirectional” maps out
//! of a Cartesian category, and describe its graphical language. In Section 2.3, we look at what happens
//! when we combine these two constructions, and the resulting graphical language of “parametric
//! lenses”. In Section 2.4, we review Cartesian reverse differential categories, a setting for categories
//! equipped with an abstract gradient operator, and how their structure relates to categories of lenses
//! and parameteric lenses.
//! Then, in the following sections, we will see how these components fit together, allowing us to
//! describe parametrised models and the algorithms used to train them.
//! 2.1
//!
//! Parametrized Maps
//!
//! In supervised learning one is typically interested in approximating a function
//! 𝑔 : R𝑛 → R𝑚
//! for some 𝑛 and 𝑚. To do this, one begins by building a neural network, which is a smooth map
//! 𝑓 : R𝑝 × R𝑛 → R𝑚
//! where R𝑝 is the set of possible weights of that neural network. Then one looks for a value of 𝑝 ∈ R𝑝
//! such that the function 𝑓 (𝑝, −) : R𝑛 → R𝑚 closely approximates 𝑔. The first thing we need to is
//! formalize these types of maps categorically, and this is done via the Para construction [Capucci
//! et al. 2021; Fong et al. 2017; Gavranovic 2019].
//! Definition 2.1 (Parametrised category). If C is a strict3 symmetric monoidal category (with
//! monoidal product ⊗ and monoidal unit 𝐼 ) then we define a category Para(C) with
//! • objects those of C
//! • a map from 𝐴 to 𝐵 in Para(C) is a pair (𝑃, 𝑓 ) where 𝑃 is an object of C and 𝑓 : 𝑃 ⊗ 𝐴 → 𝐵
//! • the identity on 𝐴 is the pair (𝐼, 1𝐴 ) (since ⊗ is strict monoidal, 𝐼 ⊗ 𝐴 = 𝐴)
//! • the composite of (𝑃, 𝑓 ) : 𝐴 → 𝐵 with (𝑃 ′, 𝑓 ′) : 𝐵 → 𝐶 is the pair
//! (𝑃 ′ ⊗ 𝑃, (1 ⊗ 𝑓 ); 𝑔).
//! Example 2.2. Our primary example for the above construction is the category Smooth whose
//! objects are natural numbers with a map 𝑓 : 𝑛 → 𝑚 a smooth map from R𝑛 to R𝑚 . As described
//! above, the category Para(Smooth) can be thought of as a category of neural networks: a map in this
//! category from 𝑛 to 𝑚 consists of a choice of 𝑝 and a map 𝑓 : R𝑝 × R𝑛 → R𝑚 with R𝑝 representing
//! the set of possible weights of the neural network.
//! As anticipated in the introduction, we represent the morphisms of Para(C) graphically using
//! the formalism of string diagrams —see [Selinger 2010] for a general overview. As we will see in the
//! next sections, the interplay of the various components at work in the learning process becomes
//! much clearer once represented with this pictorial notation.
//! In fact, we will mildly massage the traditional notation for string diagrams, which would represent
//! a morphism in Para(C) from 𝐴 to 𝐵 as below left.
//! 𝑃
//! 𝑃
//! 𝑓
//!
//! 𝐵
//!
//! 𝐴
//!
//! 𝑓
//!
//! 𝐵
//!
//! 𝐴
//!
//! Note the standard notation does not emphasise the special role played by 𝑃, which is part of the
//! data of the morphism itself. Parameters and data in machine learning have different semantics: by
//! separating them on two different axes, we obtain a graphical language which is more closely tied to
//! 3 One can also define Para( C) in the case when C is non-strict; however, the result would be not a category but a bicategory.
//!
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! Categorical Foundations of Gradient-Based Learning
//!
//! 1:7
//!
//! these semantics. Thus, we will use a slightly different convention for Para(C), writing a morphism
//! (𝑃, 𝑓 ) : 𝐴 → 𝐵 as on the right above. Incidentally, this clarifies why composition in Para(C) is
//! defined the way it is: the composite of (𝑃, 𝑓 ) : 𝐴 → 𝐵 with (𝑃 ′, 𝑓 ′) : 𝐵 → 𝐶 is simply given by
//! hooking up the 𝐵 wires:
//! 𝑃′
//!
//! 𝑃
//!
//! 𝐵
//! 𝑓′
//!
//! 𝑓
//!
//! 𝐴
//!
//! (1)
//!
//! 𝐶
//!
//! This notation also yields a neat visualisation of “reparameterisation”, as defined below.
//! Definition 2.3. A reparametrisation of (𝑃, 𝑓 ) : 𝐴 → 𝐵 in Para(C) by a map 𝛼 : 𝑄 → 𝑃 (below
//! left) is the Para(C) map (𝑄, (𝛼 ⊗ 1𝐴 ); 𝑓 ) : 𝐴 → 𝐵 (represented below right).
//! 𝑄
//! 𝛼
//! 𝑃
//! 𝑃
//! 𝑓
//!
//! 𝐴
//!
//! 𝐵
//!
//! 𝑓
//!
//! 𝐴
//!
//! (2)
//!
//! 𝐵
//!
//! Intuitively, reparameterisation changes the parameter space of (𝑃, 𝑓 ) : 𝐴 → 𝐵 to some other
//! object 𝑄, via some map 𝛼 : 𝑄 → 𝑃. We shall see later that gradient descent and its many variants
//! can naturally be viewed as reparametrisations.
//! Note coherence rules in combining operations (1) and (2) just work as expected, as these diagrams
//! can be ultimately ‘compiled’ down to string diagrams for monoidal categories. For example, given
//! maps (𝑃, 𝑓 ) : 𝐴 → 𝐵, (𝑄, 𝑔) : 𝐵 → 𝐶 with reparametrisations 𝛼 : 𝑃 ′ → 𝑃, 𝛽 : 𝑄 ′ → 𝑄, one could
//! either first reparametrise 𝑓 and 𝑔 separately and then compose the results (below left), or compose
//! first then reparametrise jointly (below right):
//! 𝑄′
//!
//! 𝑃′
//! 𝛼
//!
//! 𝑄′
//!
//! 𝛼
//!
//! 𝛽
//!
//! 𝑃
//!
//! 𝑄
//!
//! 𝑓
//!
//! 𝑔
//!
//! 𝛽
//! #
//!
//! 𝑃
//! 𝑓
//!
//! 𝐴
//!
//! 𝑃′
//!
//! 𝑄
//!
//! 𝐵
//!
//! 𝑔
//!
//! 𝐵
//!
//! 𝐶
//! 𝐴
//!
//! (3)
//! 𝐶
//!
//! As expected, translating these two operations into string diagrams for monoidal categories yield
//! equivalent representations of the same morphism.
//! 𝑄
//!
//! 𝑄′
//! 𝑃′
//!
//! 𝛽
//! 𝑔
//!
//! 𝑃
//! 𝛼
//!
//! =
//!
//! 𝑓
//! 𝐴
//!
//! 𝐶
//!
//! 𝐵
//!
//! 𝑄
//!
//! 𝑄′
//!
//! 𝛽
//!
//! 𝑃′
//!
//! 𝛼
//!
//! 𝑔
//!
//! 𝑃
//!
//! (4)
//!
//! 𝑓
//! 𝐴
//!
//! 𝐶
//!
//! 𝐵
//!
//! Remark 2.1. There is a 2-categorical perspective on Para(C), which we glossed over in this paper
//! for the sake of simplicity. In particular, the reparametrisations described above can also be seen as
//! equipping Para(C) with 2-cells, giving a 2-categorical structure on Para(C). This is also coherent
//! with respect to base change: if C and D are strict symmetric monoidal categories, and 𝐹 : C → D a
//! lax symmetric monoidal functor, then there is an induced 2-functor Para(𝐹 ) : Para(C) → Para(D)
//! which agrees with 𝐹 on objects. This 2-functor is straightforward: for a 1-cell (𝑃, 𝑓 ) : 𝐴 → 𝐵, it applies
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! 1:8
//!
//! G.S.H. Cruttwell, Bruno Gavranović, Neil Ghani, Paul Wilson, and Fabio Zanasi
//!
//! 𝐹 to 𝑃 and 𝑓 and uses the (lax) comparison to get a map of the correct type. We will see how this base
//! change becomes important when performing backpropagation on parameterised maps (Eq. 8)
//! Lastly, we mention that Para(C) inherits the symmetric monoidal structure from C and that the
//! induced 2-functor Para(𝐹 ) respects that structure. This will allow us to compose neural networks not
//! only in series, but also in parallel. For more detail on alternative viewpoints on the Para construction,
//! including how it can be viewed as the Grothendieck construction of a certain indexed category, see
//! [Capucci et al. 2021].
//! 2.2
//!
//! Lenses
//!
//! We next consider a very different categorical construction. In machine learning (or even learning in
//! general) it is fundamental that information flows both forwards and backwards: the ‘forward’ flow
//! corresponds to a model’s predictions, and the ‘backwards’ flow to corrections to the model. The
//! category of lenses is the ideal setting to capture this type of structure, as it is a category consisting
//! of maps with both a “forward” and a “backward” part.
//! Definition 2.4. For any Cartesian category C, the category of lenses4 in C, Lens(C), is the
//! category with the following data:
//! • Its objects are pairs (𝐴, 𝐴 ′), where both 𝐴 and 𝐴 ′ are objects in C
//! • A map from (𝐴, 𝐴 ′) to (𝐵, 𝐵 ′) consists of a pair (𝑓 , 𝑓 ∗ ) where 𝑓 : 𝐴 → 𝐵 (called the get or
//! forward part of the lens) and 𝑓 ∗ : 𝐴 × 𝐵 ′ → 𝐴 ′ (called the put or backwards part of the
//! lens)
//! • the composite of (𝑓 , 𝑓 ∗ ) : (𝐴, 𝐴 ′) → (𝐵, 𝐵 ′) and (𝑔, 𝑔∗ ) : (𝐵, 𝐵 ′) → (𝐶, 𝐶 ′) is given by get
//! 𝑓 ; 𝑔 and put
//! ⟨𝜋 0, ⟨𝜋0 ; 𝑓 , 𝜋1 ⟩; 𝑔∗ ⟩; 𝑓 ∗
//! • The identity on (𝐴, 𝐴 ′) is the pair (1𝐴 , 𝜋1 ).
//! It is much easier to visualize the morphisms of Lens(C) and their composites with a graphical
//! calculus as described in [Boisseau 2020, Thm. 23]. In this language, a morphism (𝑓 , 𝑓 ∗ ) : (𝐴, 𝐴 ′) →
//! (𝐵, 𝐵 ′) is written as
//!
//! 𝑓
//!
//! 𝐵
//!
//! 𝐴
//!
//! 𝐴′
//!
//! 𝑓∗
//!
//! 𝐵′
//!
//! where
//! is the string diagram (‘built-in’ in any cartesian category) duplicating the value
//! 𝐴. It is clear in this language how to describe the composite of (𝑓 , 𝑓 ∗ ) : (𝐴, 𝐴 ′) → (𝐵, 𝐵 ′) and
//! (𝑔, 𝑔∗ ) : (𝐵, 𝐵 ′) → (𝐶, 𝐶 ′): simply join the 𝐵/𝐵 ′ wires together to get the composite lens
//!
//! 4 These lenses are often called bimorphic [Hedges 2018], in contrast to simple lenses whose objects are all of the form (𝐴, 𝐴):
//!
//! the forward and backward types of simple lenses are the same.
//!
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! Categorical Foundations of Gradient-Based Learning
//!
//! 𝑓
//!
//! 1:9
//!
//! 𝑔
//!
//! 𝐵
//!
//! 𝐶
//!
//! 𝐴
//!
//! 𝐴′
//!
//! 𝑔∗
//!
//! 𝑓∗
//! 𝐵′
//!
//! 𝐶′
//!
//! and the formula for the composite in terms of equations (as described above) follows from this.
//! We will often write lenses without the inside wires exposed, thinking of the entire lens as a
//! black-box:
//! 𝐴
//! 𝐴′
//!
//! (𝑓 , 𝑓 ∗ )
//!
//! 𝐵
//! 𝐵′
//!
//! Note Lens(C) is a monoidal category, with (𝐴, 𝐴 ′) ⊗ (𝐵, 𝐵 ′) defined as (𝐴 × 𝐵, 𝐴 ′ × 𝐵 ′). However,
//! in general Lens(C) is not itself Cartesian. This is easy to see when looking at even a terminal
//! object: if 𝑇 is a terminal object in C, then in general (𝑇 ,𝑇 ) will not be a terminal object in Lens(C)
//! — it if was, there would be a unique lens (!𝐴 , !𝐴∗ ) : (𝐴, 𝐴 ′) → (𝑇 ,𝑇 ) whose put part would need to
//! be a (unique) map 𝐴 × 𝑇 → 𝐴 ′, but in general there are many such maps.
//! 2.3
//!
//! Parametric Lenses
//!
//! The fundamental category where supervised learning takes place is the composite of the two
//! constructions in the previous sections. As noted in the previous section, for a Cartesian category C,
//! Lens(C) is monoidal, and so we can form the the category Para(Lens(C)), which we shall call the
//! category of parametric lenses of C. The definition of this category follows automatically from
//! definitions of Para and Lens(C):
//! Definition 2.5. The category Para(Lens(C)) of parametric lenses on C is defined as follows.
//! • An object is a pair of objects (𝐴, 𝐴 ′) from C
//! • A morphism from (𝐴, 𝐴 ′) to (𝐵, 𝐵 ′), called a parametric lens5 , is a choice of parameter pair
//! (𝑃, 𝑃 ′) and a lens
//! (𝑓 , 𝑓 ∗ ) : (𝐴, 𝐴 ′) × (𝑃, 𝑃 ′) → (𝐵, 𝐵 ′)
//! so that 𝑓 : 𝑃 × 𝐴 → 𝐵 and 𝑓 ∗ : 𝑃 × 𝐴 × 𝐵 ′ → 𝑃 ′ × 𝐴 ′
//! By the previous two sections, we get a graphical language for Para(Lens(C)) which uses the
//! graphical language for Lens(C) from section 2.2 as a base, then augments it with parameters as
//! described in section 2.1. Thus a morphism of Para(Lens(C)) from (𝐴, 𝐴 ′) to (𝐵, 𝐵 ′) is a box with
//! input/output of (𝐴, 𝐴 ′) on the left, input/output of (𝐵, 𝐵 ′) on the right, and input/output of (𝑃, 𝑃 ′)
//! (the parameter space) on top:
//! 𝑃 𝑃′
//! 𝐴
//! 𝐴′
//!
//! 𝐵
//! 𝐵′
//!
//! (5)
//!
//! Composition is again quite natural in this formulation: given one box with input/output wires
//! (𝐵, 𝐵 ′) on the right and another box with input/output wires (𝐵, 𝐵 ′) on the left, one simply hooks
//! 5 In [Fong et al. 2017], these are called learners. However, in this paper we study them in a much broader light; see section 6.
//!
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! 1:10
//!
//! G.S.H. Cruttwell, Bruno Gavranović, Neil Ghani, Paul Wilson, and Fabio Zanasi
//!
//! up those input/output wires to get the desired composite:
//! 𝑄 𝑄′
//!
//! 𝑃 𝑃′
//! 𝐵
//! 𝐴
//! 𝐴′
//!
//! 𝐶
//! 𝐶′
//!
//! 𝐵′
//!
//! (6)
//!
//! A reparameterisation in Para(Lens(C)) is depicted graphically as drawing a box on top of the
//! (𝑃, 𝑃 ′) wires:
//! 𝑄 𝑄′
//!
//! 𝑃
//!
//! 𝑃′
//!
//! 𝐴
//! 𝐴′
//!
//! 𝐵
//! 𝐵′
//!
//! (7)
//!
//! Given a generic morphism 𝑓 in Para(Lens(C)) as depicted in (5), one can see how it is possible to
//! “learn” new values from 𝑓 : it takes as input an input 𝐴, a parameter 𝑃, and a change 𝐵 ′, and outputs
//! a change in 𝐴, a value of 𝐵, and a change 𝑃 ′. This last element is the key component for supervised
//! learning: intuitively, it says how to change the parameter values to get the neural network closer
//! to the true value of the desired function.
//! The question, then, is how one is to define such a parametric lens given nothing more than
//! a neural network, ie., a parameterized map (𝑃, 𝑓 ) : 𝐴 → 𝐵. This is precisely what the gradient
//! operation provides, and its generality to categories is explored in the next subsection.
//! 2.4
//!
//! Cartesian Reverse Differential Categories
//!
//! Fundamental to all gradient-based learning is, of course, the gradient operation. In most cases
//! this gradient operation is performed in the category of smooth maps between Euclidean spaces.
//! However, recent work [Wilson and Zanasi 2020] has shown that gradient-based learning can also
//! work well in other categories; for example, in a category of boolean circuits. Thus, to encompass
//! these examples in a single framework, it is helpful to work in a category with an abstract gradient
//! operation. Specifically, we will work in a Cartesian reverse differential category (first defined in
//! [Cockett et al. 2019]), a category in which every map has an associated reverse derivative.
//! Definition 2.6.
//! • [Cockett et al. 2019, Defn. 1] A Cartesian left additive category consists of a category C
//! with chosen finite products (including a terminal object), and an addition operation and zero
//! morphism in each homset, satisfying various axioms.
//! • [Cockett et al. 2019, Defn. 13] A Cartesian reverse differential category (CRDC) consists
//! of a Cartesian left additive category C, together with an operation which provides, for each
//! map 𝑓 : 𝐴 → 𝐵 in C, a map
//! 𝑅 [𝑓 ] : 𝐴 × 𝐵 → 𝐴
//! satisfying seven axioms (for full details, see the appendix).
//! Why are reverse derivatives helpful for learning? For 𝑓 : 𝐴 → 𝐵, the pair (𝑓 , 𝑅 [𝑓 ]) form a lens
//! from (𝐴, 𝐴) to (𝐵, 𝐵), with 𝑅 [𝑓 ] acting as backwards map. Thus having a reverse derivative already
//! provides a way to turn an ordinary map in the category into which one can pass information
//! backwards, that is, a map which can “learn”.
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! Categorical Foundations of Gradient-Based Learning
//!
//! 1:11
//!
//! Note assigning type 𝐴 × 𝐵 → 𝐴 to 𝑅 [𝑓 ] hides some relevant information: 𝐵-values in the domain
//! and 𝐴-values in the codomain of 𝑅 [𝑓 ] do not play the same role as values of the same types in
//! 𝑓 : 𝐴 → 𝐵: in 𝑅 [𝑓 ], they really take in a tangent vector at 𝐵 and output a tangent vector at 𝐴 (cf. the
//! definition of 𝑅 [𝑓 ] in Smooth, Example 2.8 below). To emphasise this, we will type 𝑅 [𝑓 ] as a map
//! 𝐴 × 𝐵 ′ → 𝐴 ′ (even though in reality 𝐴 = 𝐴 ′ and 𝐵 = 𝐵 ′), thus meaning that (𝑓 , 𝑅 [𝑓 ]) is actually a
//! lens from (𝐴, 𝐴 ′) to (𝐵, 𝐵 ′). This typing distinction will be helpful later on, when we want to add
//! additional components to our learning algorithms.
//! Graphically, then, we represent the pair (𝑓 , 𝑅 [𝑓 ]) as a lens:
//!
//! 𝑓
//!
//! 𝐵
//!
//! 𝐴
//!
//! 𝐴′
//!
//! 𝑅 [𝑓 ]
//!
//! 𝐵′
//!
//! This point of view also makes clear the usefulness of the reverse chain rule (axiom [RD.5] from
//! [Cockett et al. 2019, Defn. 13]): it tells us that the operation which takes a map 𝑓 and produces the
//! lens (𝑓 , 𝑅 [𝑓 ]) preserves composition: that is,
//!
//! 𝑓𝑔
//!
//! 𝑓
//!
//! 𝐶
//!
//! 𝐴
//!
//! 𝐵
//!
//! 𝑔
//!
//! 𝐶
//!
//! 𝐴
//!
//! =
//! 𝐴′
//!
//! 𝑅 [𝑓 𝑔]
//!
//! 𝐶′
//!
//! 𝐴′
//!
//! 𝑅 [𝑓 ]
//!
//! 𝑅 [𝑔]
//! 𝐵′
//!
//! 𝐶′
//!
//! Combined with axiom [RD.3] for a CRDC, this justifies the following fact, which we record for
//! later use.
//! Proposition 2.7. [Cockett et al. 2019, Prop. 31] If C is a CRDC, there is a functor R : C → Lens(C)
//! which on objects sends 𝐴 to the pair (𝐴, 𝐴) and on maps sends 𝑓 : 𝐴 → 𝐵 to the pair (𝑓 , 𝑅 [𝑓 ]).
//! The following two examples of CRDCs will serve as the basis for the learning scenarios of the
//! upcoming sections.
//! Example 2.8. The category Smooth has as objects natural numbers and maps 𝑛 → 𝑚 are m-tuples
//! of smooth maps 𝑓 : R𝑛 → R. Smooth is Cartesian with product given by addition. Smooth is a
//! Cartesian reverse differential category: given a smooth map 𝑓 : R𝑛 → R𝑚 , the map
//! 𝑅 [𝑓 ] : R𝑛 × R𝑚 → R𝑛
//! sends a pair (𝑥, 𝑣) to 𝐽 [𝑓 ]𝑇 (𝑥) · 𝑣: the transpose of the Jacobian of 𝑓 at 𝑥 in the direction 𝑣.
//! For example, if 𝑓 : R2 → R3 is defined as 𝑓 (𝑥 1, 𝑥 2 ) := (𝑥 13 + 2𝑥 1𝑥 2, 𝑥 2, sin(𝑥 1 )), then 𝑅 [𝑓 ] :
//! 
//!  𝑣 
//! 3𝑥 1 + 2𝑥 2 0 cos(𝑥 1 )  1 
//! 2
//! 3
//! 2
//! R × R → R is given by (𝑥, 𝑣) ↦→
//! · 𝑣 2  . Using the reverse derivative
//! 2𝑥 1
//! 1
//! 0
//! 𝑣 3 
//!  
//! 6
//! (as opposed to the forward derivative ) is well-known to be much more computationally efficient
//! 6 Forward derivatives can analogously be modelled with Cartesian (forward) differential categories [Cruttwell 2012]. Given
//!
//! a map 𝑓 : 𝐴 → 𝐵 the forward derivative 𝐷 [𝑓 ] : 𝐴 × 𝐴 → 𝐵 uses the Jacobian of 𝑓 , instead of its transpose.
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! 1:12
//!
//! G.S.H. Cruttwell, Bruno Gavranović, Neil Ghani, Paul Wilson, and Fabio Zanasi
//!
//! for functions 𝑓 : R𝑛 → R𝑚 when 𝑚 ≪ 𝑛 (for example, see [Griewank and Walther 2008]), as is the
//! case in most supervised learning situations (where often 𝑚 = 1).
//! Example 2.9. Another RDC is the PROP POLYZ2 [Cockett et al. 2019, Example 14], whose morphisms 𝑓 : 𝐴 → 𝐵 are 𝐵-tuples of polynomials Z2 [𝑥 1 . . . 𝑥𝐴 ]. When presented by generators and
//! relations these morphisms can be viewed as a syntax for boolean circuits, with parametric lenses
//! for such circuits (and their reverse derivative) described in [Wilson and Zanasi 2020].
//! 3
//!
//! COMPONENTS OF LEARNING AS PARAMETRIC LENSES
//!
//! As seen in the introduction, in the learning process there are many components at work: a model,
//! an optimiser, a loss map, a learning rate, etc. In this section we show how each such component
//! can be understood as a parametric lens. Moreover, for each component, we show how our framework encompasses several variations of the gradient-descent algorithms, thus offering a unifying
//! perspective on many different approaches that appear in the literature.
//! 3.1
//!
//! Models as Parametric Lenses
//!
//! We begin by characterising the models used for training as a parametric lenses. In essence, our
//! approach identifies a set of abstract requirements necessary to perform training by gradient descent,
//! which covers the case studies that we will consider in the next sections.
//! The leading intuition is that a suitable model is a parametrised map, equipped with a reverse
//! derivative operator. Using the formal developments of Section 2, this amounts to assuming that a
//! model is a morphism in Para(C), for a CRDC C. In order to visualise such morphism as a parametric
//! lens, it then suffices to apply R from Proposition 2.7 under Para(−), yielding a functor
//! Para(R) : Para(C) → Para(Lens(C))
//!
//! (8)
//!
//! Pictorially, Para(R) send a map as on the left below to a parametric lens as on the right.
//! 𝑃
//!
//! 𝑃′
//!
//! 𝑓
//!
//! 𝐵
//!
//! 𝑃
//! 𝐴
//! 𝐴
//!
//! 𝑓
//!
//! 𝐵
//!
//! ↦→
//! 𝐴′
//!
//! 𝑅 [𝑓 ]
//!
//! 𝐵′
//!
//! Example 3.1 (Neural networks). As noted previously, to learn a function of type R𝑛 → R𝑚 , one
//! constructs a neural network, which can be seen as a function of type R𝑝 × R𝑛 → R𝑚 where R𝑝 is
//! the space of parameters of the neural network. As seen in Example 2.2, this is a map in the category
//! Para(Smooth) of type R𝑛 → R𝑚 with parameter space R𝑝 . Then one can apply the functor in (8)
//! to present a neural network together with its reverse derivative operator as a parameterised lens,
//! i.e. a morphism in Para(Lens(Smooth)).
//! Example 3.2 (Boolean circuits). For learning of Boolean circuits as described in [Wilson and
//! Zanasi 2020], almost everything is the same as the previous example except that the base category
//! is changed to polynomials over Z2 , POLYZ2 . To learn a function of type Z𝑛2 → Z𝑚
//! 2 , one constructs
//! some map of type Z𝑝 × Z𝑛 → Z𝑚 , which one can view as a map in the category Para(POLYZ2 ) of
//! type Z𝑛 → Z𝑚 with parameter space Z𝑝 , and again applying the functor in (8) for POLYZ2 yields a
//! parameterised lens.
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! Categorical Foundations of Gradient-Based Learning
//!
//! 1:13
//!
//! Note a model/parametric lens 𝑓 can take as inputs an element of 𝐴, an element of 𝐵 ′ (a change in
//! 𝐵) and a parameter 𝑃 and outputs an element of 𝐵, a change in 𝐴, and a change in 𝑃. This is not yet
//! sufficient to do machine learning! When we perform learning, we want to input a parameter 𝑃 and
//! a pair 𝐴 × 𝐵 and receive a new parameter 𝑃. Instead, 𝑓 expects a change in 𝐵 (not an element of 𝐵)
//! and outputs a change in 𝑃 (not an element of 𝑃). Deep dreaming, on the other hand, wants to return
//! an element of 𝐴 (not a change in 𝐴). Thus, to do machine learning (or deep dreaming) we need to
//! add additional components to 𝑓 ; we will consider these additional components in the next sections.
//! 3.2
//!
//! Loss Maps as Parametric Lenses
//!
//! Another key component of any learning algorithm is the choice of loss map. This gives a measurement of how far the current output of the model is from the desired output. In standard learning
//! in Smooth, this loss map is viewed as a map of type 𝐵 × 𝐵 → R. However, in our setup, this is
//! naturally viewed as a parameterised map from 𝐵 to R with parameter space 𝐵.7 We also generalize
//! the codomain to an arbitrary object 𝐿.
//! Definition 3.3. A loss map on 𝐵 consists of a Para(C) map (loss, 𝐵) : 𝐵 → 𝐿 for some object 𝐿.
//! We can then compose such a map with a neural network (𝑁 , 𝑃) : 𝐴 → 𝐵 to get the composite
//! 𝑃 𝑃′
//!
//! 𝐵 𝐵′
//! 𝐵
//!
//! 𝐴
//! 𝐴′
//!
//! loss
//!
//! 𝑁
//! 𝐵′
//!
//! 𝐿
//! 𝐿′
//!
//! (9)
//!
//! Note we can apply Para to either composite 𝑁 𝐿 or 𝑁 and 𝐿 individually, giving a parametric lens
//! 𝑃
//!
//! 𝐵
//!
//! 𝑃
//!
//! 𝐵
//!
//! Compose
//! 𝐵
//! 𝑓
//!
//! 𝐴
//!
//! 𝐵 𝐵
//!
//! loss
//!
//! 𝐿
//!
//! loss
//!
//! 𝑓
//!
//! 𝐴
//!
//! Para(𝐹 )
//!
//! 𝐿
//!
//! Para(𝐹 )
//!
//! (10)
//! 𝑃
//!
//! 𝑃′
//!
//! 𝐵
//!
//! 𝐵′
//!
//! 𝑃
//!
//! 𝑃′
//!
//! 𝐵 𝐵
//! 𝐴
//! 𝐴′
//!
//! 𝑓
//! 𝑅 [𝑓 ]
//!
//! 𝐵′
//!
//! 𝐵′
//!
//! 𝐵
//!
//! 𝐵′
//!
//! 𝐵
//! loss
//! 𝑅 [loss]
//!
//! 𝐿
//! 𝐿′
//!
//! 𝐴
//! 𝐴′
//! Compose
//!
//! 𝑓
//! 𝑅 [𝑓 ]
//!
//! 𝐵′
//!
//! loss
//! 𝑅 [loss]
//!
//! 𝐿
//! 𝐿′
//!
//! This is getting closer to the parametric lens we want: it can now receive inputs of type 𝐵. However,
//! this is at the cost of now needing an input to 𝐿 ′; we consider how to handle this in the next section.
//! Example 3.4 (Quadratic error). In Smooth, the standard loss function on R𝑏 is quadratic error: it
//! uses 𝐿 = R and has parameterised map 𝑒 : R𝑏 × R𝑏 → R given by
//! 𝑏
//!
//! 𝑒 (𝑏𝑡 , 𝑏𝑝 ) =
//!
//! 1 ∑︁
//! ((𝑏 𝑝 )𝑖 − (𝑏𝑡 )𝑖 ) 2 .
//! 2 𝑖=1
//!
//! 7 Here the loss map has its parameter space equal to its input space. However, putting loss maps on the same footing as
//!
//! models lends itself to further generalizations where the parameter space is different, and where the loss map can itself be
//! learned. See Generative Adversarial Networks, [Capucci et al. 2021, Figure 7.].
//!
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! 1:14
//!
//! G.S.H. Cruttwell, Bruno Gavranović, Neil Ghani, Paul Wilson, and Fabio Zanasi
//!
//! where we think of 𝑏𝑡 as the “true” value and 𝑏 𝑝 the predicted value. This has reverse derivative
//! 𝑅 [𝑒] : R𝑏 × R𝑏 × R → R𝑏 × R𝑏 given by8 𝑅 [𝑒] (𝑏𝑡 , 𝑏𝑝 , 𝛼) = 𝛼 · (𝑏 𝑝 − 𝑏𝑡 , 𝑏𝑡 − 𝑏 𝑝 ).
//! Example 3.5 (Boolean error). In POLYZ2 , the loss function on Z𝑏 which is implictly used in [Wilson
//! and Zanasi 2020] is a bit different: it uses 𝐿 = Z𝑏 and has parameterised map 𝑒 : Z𝑏 × Z𝑏 → Z𝑏
//! given by
//! 𝑒 (𝑏𝑡 , 𝑏 𝑝 ) = 𝑏𝑡 + 𝑏 𝑝 .
//! (Note that this is + in 𝑍 2 ; equivalently this is given by XOR.) Its reverse derivative is of type
//! 𝑅 [𝑒] : Z𝑏 × Z𝑏 × Z𝑏 → Z𝑏 × Z𝑏 given by 𝑅 [𝑒] (𝑏𝑡 , 𝑏 𝑝 , 𝛼) = (𝛼, 𝛼).
//! Example 3.6 (Softmax cross entropy). The Softmax cross entropy loss is a R𝑏 -parameterized map
//! R → R defined by
//! ∑︁
//! 𝑒 (𝑏𝑡 , 𝑏 𝑝 ) =
//! (𝑏𝑡 )𝑖 ((𝑏 𝑝 )𝑖 − log(Softmax(𝑏 𝑝 )𝑖 )
//! 𝑏
//!
//! 𝑖:𝑁
//!
//! where Softmax(𝑏 𝑝 ) = Í
//!
//! exp( (𝑏𝑝 )𝑖 )
//! is defined componentwise for 𝑖 : 𝑁 .
//! 𝑗 :𝑁 exp( (𝑏 𝑝 ) 𝑗 )
//!
//! We note that, although 𝑞 needs to be a probability distribution, at the moment there is no
//! need to ponder the question of interaction of probability distributions with the reverse derivative
//! framework: one can simply consider 𝑞 as the image of some logits under the Softmax function.
//! Example 3.7 (Dot product). In Deep Dreaming (Section 4.2) we often want to focus only on a
//! particular element of the network output R𝑏 . This is done by supplying a one-hot vector 𝑏𝑡 as the
//! ground truth to the loss function which takes two vectors and computes their dot product:
//! 𝑒 (𝑏𝑡 , 𝑏 𝑝 ) = 𝑏𝑡 · 𝑏 𝑝
//! The reverse derivative of the dot product has the type R𝑏 × R𝑏 × R → R𝑏 × R𝑏 and the following
//! implementation
//! 𝑅 [𝑒] (𝑏𝑡 , 𝑏 𝑝 , 𝛼) = (𝛼 · 𝑏 𝑝 , 𝛼 · 𝑏𝑡 )
//! If the ground truth vector 𝑦 is a one-hot vector (active at the 𝑖-th element), then the dot product
//! essentially performs masking of all inputs except the 𝑖-th one.
//! 3.3
//!
//! Learning Rates as Parametric Lenses
//!
//! With a loss function, we are getting closer to our goal of having a parameterised lens which
//! represents a learning process. We have the following parametrised lens:
//! 𝑃 𝑃′
//!
//! 𝐵 𝐵′
//! 𝐵
//!
//! 𝐴
//! 𝐴′
//!
//! 𝑓
//! 𝑅 [𝑓 ]
//!
//! 𝐵′
//!
//! loss
//! 𝑅 [loss]
//!
//! 𝐿
//! 𝐿′
//!
//! (11)
//!
//! In this section we focus on the right side of the diagram. There is an output of 𝐿 and an input
//! of 𝐿 ′ to the diagram. This is precisely the place where gradient-based learning algorithms input a
//! learning rate.
//! Definition 3.8. A learning rate 𝛼 on 𝐿 consists of a lens from (𝐿, 𝐿 ′) to (1, 1) where 1 is a terminal
//! object in C.
//! 8 The last argument in the reverse derivative implementation is suggestively named 𝛼 to envoke the idea of learning rate,
//!
//! described in the next subsection.
//!
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! Categorical Foundations of Gradient-Based Learning
//!
//! 1:15
//!
//! Note that the get component of such a lens must be the unique map to 1, while the put component
//! is a map 𝐿 ×1 → 𝐿 ′; that is, simply a map 𝛼 ∗ : 𝐿 → 𝐿 ′. Moreover, we can view 𝛼 as a Para(Lens(C))
//! map from (𝐿, 𝐿 ′) → (1, 1) (with trivial parameter space). We write such a morphism as a cap, and
//! compose it with the parameterised map above to get
//! 𝑃 𝑃′
//!
//! 𝐵 𝐵′
//! 𝐵
//!
//! 𝐴
//! 𝐴′
//!
//! 𝑓
//! 𝑅 [𝑓 ]
//!
//! 𝐵′
//!
//! 𝐿
//! loss
//! 𝑅 [loss]
//!
//! 𝛼
//! 𝐿′
//!
//! Fig. 3. Model composed with a loss function and a learning rate
//!
//! Example 3.9. In standard supervised learning in Smooth, one fixes some 𝜖 > 0 as a learning rate,
//! and this is used to define 𝛼: 𝛼 is simply constantly −𝜖, ie., 𝛼 (𝑙) = −𝜖 for any 𝑙 ∈ 𝐿.
//! Example 3.10. In supervised learning in POLYZ2 , the standard learning rate is quite different: for
//! a given 𝐿 it is defined as the identity function, 𝛼 (𝑙) = 𝑙.
//! Other learning rate morphisms are possible as well: for example, one could fix some 𝜖 > 0 and
//! define a learning rate in Smooth by 𝛼 (𝑙) = −𝜖 · 𝑙. Such a learning rate would take into account how
//! far away the network is from its desired goal and adjust its learning rate accordingly.
//! 3.4
//!
//! Optimisers as Reparameterisations
//!
//! In the previous sections we have seen how to incorporate the loss map and learning rate into our
//! formalism. In this section we consider how to implement gradient descent (and its variants) into the
//! 𝑃 𝑃′
//!
//! picture. Recall that we are writing our model graphically as the parameterised lens 𝐴′
//! 𝐴
//!
//! 𝐵
//! .
//! 𝐵′
//!
//! Note that this diagram outputs a 𝑃 ′, which represents a change in the parameter space. But we
//! would like to receive not just the requested change in the parameter, but the new parameter itself.
//! Thus, we need to add a box above the 𝑃/𝑃 ′ wires in the image above to get something whose input
//! and output are both 𝑃. Recall that a box in the graphical language is a lens; thus, we are asking for
//! a lens of type (𝑃, 𝑃) → (𝑃, 𝑃 ′). This is precisely what gradient descent accomplishes.
//! Definition 3.11. In any CRDC C we can define gradient update as a map 𝐺 in Lens(C) from
//! (𝑃, 𝑃) to (𝑃, 𝑃 ′) consisting of
//! (𝐺, 𝐺 ∗ ) : (𝑃, 𝑃) → (𝑃, 𝑃 ′)
//! where 𝐺 (𝑝) = 𝑝 and 𝐺 ∗ (𝑝, 𝑝 ′) = 𝑝 + 𝑝 ′.
//! Note that gradient descent is not typically seen as a lens - but it precisely fits this way into the
//! picture we are creating!
//! Gradient descent allows one to receive the requested change in parameter and implement that
//! change by adding that value to the current parameter. We attach this lens, seen as a reparameterisation, to the top of the diagram above, giving us Figure 4 (left).
//! Example 3.12 (Gradient update in Smooth). In Smooth, the gradient descent reparameterisation
//! will take the output from 𝑃 ′ and add it to the current value of 𝑃 to get a new value of 𝑃.
//! Example 3.13 (Gradient update in Boolean circuits). In the CRDC POLYZ2 , the gradient descent
//! reparameterisation will again take the output from 𝑃 ′ and add it to the current value of 𝑃 to get
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! 1:16
//!
//! G.S.H. Cruttwell, Bruno Gavranović, Neil Ghani, Paul Wilson, and Fabio Zanasi
//!
//! 𝑃
//!
//! 𝑆 ×𝑃
//!
//! 𝑃
//!
//! Optimiser
//!
//! +
//!
//! 𝑃
//!
//! 𝑃′
//!
//! 𝐴
//!
//! 𝑃′
//!
//! 𝑃
//! 𝐵
//!
//! 𝐴
//!
//! 𝐵′
//!
//! 𝐴′
//!
//! Model
//! 𝐴′
//!
//! 𝑆 ×𝑃
//!
//! 𝐵
//! Model
//! 𝐵′
//!
//! Fig. 4. Model reparameterised by basic gradient descent (left) and a generic stateful optimiser (right).
//!
//! a new value of 𝑃; however, since + in Z2 is the same as XOR, this can be also be seen as taking
//! the XOR of the current parameter and the requested change; this is exactly how this algorithm is
//! implemented in [Wilson and Zanasi 2020].
//! Moreover, other variants of gradient descent also fit naturally into this framework by allowing
//! for additional input/output data with 𝑃. In particular, many important variants of gradient descent
//! keep track of the history of previous updates and use that to inform the next one. This is easy to
//! model in our setup: instead of asking for a lens from (𝑃, 𝑃) to (𝑃, 𝑃 ′), we ask instead for a lens from
//! (𝑆 × 𝑃, 𝑆 × 𝑃) to (𝑃, 𝑃 ′) where 𝑆 is some other object which holds a “state”.
//! Definition 3.14. A stateful parameter update consists of a choice of object 𝑆 (the state object)
//! and a lens 𝑈 : (𝑆 × 𝑃, 𝑆 × 𝑃) → (𝑃, 𝑃 ′).
//! Again, we view this optimizer as a reparameterisation and attach it above the 𝑃/𝑃 ′ wires to the
//! image from the previous section, giving us Figure 4 (right). Let us consider how several well-known
//! optimizers can be implemented in this way.
//! Example 3.15 (Momentum). In the momentum variant of gradient descent, one keeps track of the
//! previous change and uses this to inform how the current parameter should be changed. Thus, in
//! this case, we set 𝑆 = 𝑃, fix some 𝛾 > 0, and define the momentum lens
//! (𝑈 , 𝑈 ∗ ) : (𝑃 × 𝑃, 𝑃 × 𝑃) → (𝑃, 𝑃 ′)
//! by 𝑈 (𝑠, 𝑝) = 𝑝 and 𝑈 ∗ (𝑠, 𝑝, 𝑝 ′) = (𝑠 ′, 𝑝 +𝑠 ′), where 𝑠 ′ = −𝛾𝑠 + 𝑝 ′. Note momentum recovers gradient
//! descent when 𝛾 = 0.
//! In both standard gradient descent and momentum, our lens representation has trivial get/forward
//! part. Thus it is reasonable to wonder whether this formulation is really capturing the essence of
//! what is going on. However, as soon as we move to more complicated variants, having non-trivial
//! forward part of the lens is important, and Nesterov momentum is a key example of this.
//! Example 3.16 (Nesterov momentum). In Nesterov momentum, one makes a modification to the
//! input of the derivative by the previous update. We can precisely capture this by using a small
//! variation of the lens in the previous example. Again, we set 𝑆 = 𝑃, fix some 𝛾 > 0, and define the
//! Nesterov momentum lens
//! (𝑈 , 𝑈 ∗ ) : (𝑃 × 𝑃, 𝑃 × 𝑃) → (𝑃, 𝑃 ′)
//! by 𝑈 (𝑠, 𝑝) = 𝑝 + 𝛾𝑠 and 𝑈 ∗ as in the previous example.
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! Categorical Foundations of Gradient-Based Learning
//!
//! 1:17
//!
//! Example 3.17 (Adagrad). Given any fixed 𝜖 > 0 and 𝛿 ∼ 10−7 , Adagrad [Duchi et al. 2011] is given
//! by 𝑆 = 𝑃, with the lens whose get part is (𝑔, 𝑝) ↦→ 𝑝. The put is (𝑔, 𝑝, 𝑝 ′) ↦→ (𝑔 ′, 𝑝 + 𝛿+𝜖√𝑔′ ⊙ 𝑝 ′)
//! where 𝑔 ′ = 𝑔 + 𝑝 ′ ⊙ 𝑝 ′ and ⊙ is the elementwise (Hadamard) product.
//! Unlike with other optimization algorithms where the learning rate is the same for all parameters,
//! Adagrad divides the learning rate of each individual parameter with the square root of the past
//! accumulated gradients.
//! Example 3.18 (Adam). Adaptive Moment Estimation (Adam) [Kingma and Ba 2015] is another
//! method that computes adaptive learning rates for each parameter by storing exponentially decaying
//! average of past gradients (𝑚) and past squared gradients (𝑣). Fixed 𝛽 1, 𝛽 2 ∈ [0, 1), 𝜖 > 0, and 𝛿 ∼ 10−8 ,
//! Adam is given by 𝑆 = 𝑃 × 𝑃, with the lens whose get part is (𝑚, 𝑣, 𝑝) ↦→ 𝑝 and whose put part is
//! put(𝑚, 𝑣, 𝑝, 𝑝 ′) = (𝑚
//! b′, b
//! 𝑣 ′, 𝑝 +
//!
//! 𝜖
//! b′ )
//! √ ⊙𝑚
//! ′
//! 𝛿+ b
//! 𝑣
//! ′
//!
//! ′
//!
//! 𝑚
//! 𝑣
//! where 𝑚 ′ = 𝛽 1𝑚 + (1 − 𝛽 1 )𝑝 ′, 𝑣 ′ = 𝛽 2𝑣 + (1 − 𝛽 2 )𝑝 ′2 , and 𝑚
//! b′ = 1−𝛽
//! 𝑣 ′ = 1−𝛽
//! 𝑡 ,b
//! 𝑡 .
//! 1
//!
//! 2
//!
//! Note that, so far, optimsers/reparameterisations have been added to the 𝑃/𝑃 ′ wires, in order to
//! change the model’s parameters. We will see in section 4.2 how we can also attach them to the 𝐴/𝐴 ′
//! wires instead, giving deep dreaming.
//! 4
//!
//! LEARNING WITH PARAMETRIC LENSES
//!
//! In the previous section we have seen how all the components of learning can be modeled as
//! parametric lenses. We now study how all these components can be put together to form supervised
//! learning systems. In addition to studying the most common examples of supervised learning:
//! systems that learn parameters, we also study different kinds systems: those that learn their inputs.
//! This is a technique commonly known as deep dreaming, and we present it as a natural counterpart
//! of supervised learning of parameters.
//! Before we describe these systems, it will be convenient to represent all the inputs and outputs of
//! our parametric lenses as parameters. In Figure 3, we see the 𝑃/𝑃 ′ and 𝐵/𝐵 ′ inputs and outputs as
//! parameters; however, the 𝐴/𝐴 ′ wires are not. To view the 𝐴/𝐴 ′ inputs as parameters, we compose
//! that system with the parameterised lens 𝜂 we now define. The parameterised lens 𝜂 has the type
//! (1, 1) → (𝐴, 𝐴 ′) with parameter space (𝐴, 𝐴 ′) defined by (get𝜂 = 1𝐴 , put𝜂 = 𝜋1 ) and can be depicted
//! 𝐴
//! 𝐴
//!
//! graphically as
//!
//! . Composing 𝜂 with the rest of the learning system in Figure 3 gives us
//! 𝐴′
//!
//! the closed parametric lens in Figure 5.
//! 𝐴
//!
//! 𝐴
//!
//! 𝑃
//!
//! 𝑃′
//!
//! 𝐴
//!
//! 𝐵
//!
//! 𝐿
//! Loss
//!
//! Model
//! 𝐴′
//!
//! 𝐵′
//!
//! 𝐵
//!
//! 𝐵′
//!
//! 𝛼
//! 𝐿′
//!
//! Fig. 5. Closed parametric lens whose all inputs and outputs are now vertical wires.
//!
//! This composite is now a map in Para(Lens(C)) from (1, 1) to (1, 1); all its inputs and outputs are
//! now vertical wires, ie., parameters. Unpacking it further, this is a lens of type (𝐴×𝑃×𝐵, 𝐴 ′ ×𝑃 ′ ×𝐵 ′) →
//! (1, 1) whose get map is the terminal map, and whose put map is of the type 𝐴 ×𝑃 × 𝐵 → 𝐴 ′ ×𝑃 ′ × 𝐵 ′.
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! 1:18
//!
//! G.S.H. Cruttwell, Bruno Gavranović, Neil Ghani, Paul Wilson, and Fabio Zanasi
//!
//! It can be unpacked as the composite
//! put(𝑎, 𝑝, 𝑏𝑡 ) = (𝑎 ′, 𝑝 ′, 𝑏𝑡′ )
//!
//! 𝑏 𝑝 = 𝑓 (𝑝, 𝑎)
//! ′ ′
//! (𝑏𝑡 , 𝑏 𝑝 ) = 𝑅 [loss] (𝑏𝑡 , 𝑏𝑝 , 𝛼 (loss(𝑏𝑡 , 𝑏 𝑝 )))
//! (𝑝 ′, 𝑎 ′) = 𝑅 [𝑓 ] (𝑝, 𝑎, 𝑏 𝑝′ )
//!
//! where
//!
//! In the next two sections we consider further additions to the image above which correspond to
//! different types of supervised learning: supervised learning of parameters and supervised learning
//! of inputs.
//! 4.1
//!
//! Supervised Learning of Parameters
//!
//! The most common type of learning that is performed on the image in Figure 5 is supervised learning
//! of parameters. This is done by reparameterising the image (Def. 2.3) in the following manner. The
//! parameter ports are reparameterised by one of the (potentially stateful) optimisers described in the
//! previous section, while the backward wires 𝐴 ′ of inputs and 𝐵 ′ of outputs are discarded. This finally
//! gives us the complete picture of a system which learns the parameters in a supervised manner
//! (Figure 6).
//! 𝑆 ×𝑃
//!
//! 𝐴
//!
//! 𝑆 ×𝑃
//!
//! 𝐵
//!
//! Optimiser
//!
//! 𝑃
//!
//! 𝑃′
//!
//! 𝐴
//!
//! 𝐵′
//! 𝐵
//! Loss
//!
//! Model
//! 𝐴′
//!
//! 𝐿
//!
//! 𝐵′
//!
//! 𝛼
//! 𝐿′
//!
//! Fig. 6. Closed parametric lens whose parameters are being learned. An animation of this supervised learning
//! system is available online. 9
//!
//! Fixing a particular optimiser (𝑈 , 𝑈 ∗ ) : (𝑆 × 𝑃, 𝑆 × 𝑃) → (𝑃, 𝑃 ′) we again unpack the entire
//! construction. This is a map in Para(Lens(C)) from (1, 1) to (1, 1) whose parameter space is (𝐴 ×
//! 𝑆 × 𝑃 × 𝐵, 𝑆 × 𝑃). In other words, this is a lens of type (𝐴 × 𝑆 × 𝑃 × 𝐵, 𝑆 × 𝑃) → (1, 1) whose get
//! component is identity. Its put map has the type 𝐴 × 𝑆 × 𝑃 × 𝐵 → 𝑆 × 𝑃 and unpacks to
//! put(𝑎, 𝑠, 𝑝, 𝑏𝑡 ) = 𝑈 ∗ (𝑠, 𝑝, 𝑝 ′)
//!
//! where
//!
//! 𝑝 = 𝑈 (𝑠, 𝑝)
//! 𝑏 𝑝 = 𝑓 (𝑝, 𝑎)
//! ′ ′
//! (𝑏𝑡 , 𝑏 𝑝 ) = 𝑅 [loss] (𝑏𝑡 , 𝑏 𝑝 , 𝛼 (loss(𝑏𝑡 , 𝑏 𝑝 )))
//! (𝑝 ′, 𝑎 ′) = 𝑅 [𝑓 ] (𝑝, 𝑎, 𝑏 𝑝′ )
//!
//! While this formulation might seem daunting, we note that it just explicitly specifies the computation performed by a supervised learning system. The variable 𝑝 represents the parameter supplied
//! to the network by the stateful gradient update rule (in many cases this is equal to 𝑝); 𝑏 𝑝 represents
//! the prediction of the network (contrast this with 𝑏𝑡 which represents the ground truth from the
//! dataset), Variables with a tick ′ represent changes: 𝑏 𝑝′ and 𝑏𝑡′ are the changes on predictions and
//! 9 See footnote 1.
//!
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! Categorical Foundations of Gradient-Based Learning
//!
//! 1:19
//!
//! true values respectively, while 𝑝 ′ and 𝑎 ′ are changes on the parameters and inputs. Furthermore,
//! this arises automatically out of the rule for lens composition (Figure 2.2); what we needed to specify
//! is just the lenses themselves.
//! We justify and illustrate our approach on a series of case studies drawn from the machine
//! learning literature, showing how in each case the parameters of our framework (in particular,
//! loss functions and gradient updates) instantiate to familiar concepts. This presentation has the
//! advantage of treating all these case studies uniformly in terms of our basic constructs, highlighting
//! their similarities and differences.
//! We start in Smooth, fixing some parameterised map (R𝑝 , 𝑓 ) : Para(Smooth) (R𝑎 , R𝑏 ) and the
//! constant negative learning rate 𝛼 : R (Example 3.9). We then vary the loss function and the gradient
//! update, seeing how the put map above reduces to many of the known cases in the literature.
//! Example 4.1 (Quadratic error, basic gradient descent). Fix the quadratic error (Example 3.4) as the
//! loss map and basic gradient update (Example 3.12). Then the aforementioned put map simplifies.
//! Since there is no state, its type reduces to 𝐴 × 𝑃 × 𝐵 → 𝑃, and its implementation to: put(𝑎, 𝑝, 𝑏𝑡 ) =
//! 𝑝 + 𝑝 ′, where (𝑝 ′, 𝑎 ′) = 𝑅 [𝑓 ] (𝑝, 𝑎, 𝛼 · (𝑓 (𝑝, 𝑎) − 𝑏𝑡 )).
//! Note that 𝛼 here is simply a constant, and due to the linearity of the reverse derivative (Def A.5),
//! we can slide the 𝛼 from the costate into the gradient descent lens. Rewriting this update, and
//! performing this sliding we obtain a closed form update step
//! put(𝑎, 𝑝, 𝑏𝑡 ) = 𝑝 + 𝛼 · (𝑅 [𝑓 ] (𝑝, 𝑎, 𝑓 (𝑝, 𝑎) − 𝑏𝑡 ); 𝜋0 )
//! where the negative descent component of gradient descent is here contained in the choice of the
//! negative constant 𝛼.
//! This example gives us a variety of regression algorithms solved iteratively by gradient descent: it
//! embeds some parameterised map (R𝑝 , 𝑓 ) : Para(Smooth) (R𝑎 , R𝑏 ) into the system which performs
//! regression on input data - where 𝑎 denotes the input to the model and 𝑏𝑡 denotes the ground truth.
//! If the corresponding map 𝑓 is linear and 𝑚 = 1, we recover simple linear regression with gradient
//! descent. If the codomain is additionally multi-dimensional, i.e. we’re predicting multiple scalars,
//! then we recover multivariate linear regression. Likewise, we can model a multi-layer perceptron or
//! even more complex neural network architectures performing supervised learning of parameters
//! simply by changing the underlying parameterised map.
//! Example 4.2 (Softmax cross entropy, basic gradient descent). Fix Softmax cross entropy (Example
//! 3.6) as the loss map and basic gradient update (Example 3.12). Again the put map simplifies. The
//! type reduces to 𝐴 × 𝑃 × 𝐵 → 𝑃 and the implementation to
//! put(𝑎, 𝑝, 𝑏𝑡 ) = 𝑝 + 𝑝 ′
//! where (𝑝 ′, 𝑎 ′) = 𝑅 [𝑓 ] (𝑝, 𝑎, 𝛼 · (Softmax(𝑓 (𝑝, 𝑎)) − 𝑏𝑡 )). The same rewriting performed on the
//! previous example can be done here.
//! This example recovers logistic regression, e.g. classification.
//! Example 4.3 (Mean squared error, Nesterov Momentum). Fix the quadratic error (Example 3.4) as
//! the loss map and Nesterov momentum (Example 3.16) as the gradient update. This time the put
//! map doesn’t have a simplified type, it is still 𝐴 × 𝑆 × 𝑃 × 𝐵 → 𝑆 × 𝑃. The implementation of put
//! reduces to
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! 1:20
//!
//! G.S.H. Cruttwell, Bruno Gavranović, Neil Ghani, Paul Wilson, and Fabio Zanasi
//!
//! put(𝑎, 𝑠, 𝑝, 𝑏𝑡 ) = (𝑠 ′, 𝑝 + 𝑠 ′)
//!
//! 𝑝 = 𝑝 + 𝛾𝑠
//!
//! where
//! ′
//!
//! ′
//!
//! (𝑝 , 𝑎 ) = 𝑅 [𝑓 ] (𝑝, 𝑎, 𝛼 · (𝑓 (𝑝, 𝑎) − 𝑏𝑡 ))
//! 𝑠 ′ = −𝛾𝑠 + 𝑝 ′
//! This example with Nesterov momentum differs in two key points from all the other ones: i) the
//! optimiser is stateful, and ii) its get map is not trivial. While many other optimisers are stateful,
//! the non-triviality of the get map here showcases the importance of lenses. They allow us to make
//! precise the notion of computing a “lookahead” value for Nesterov momentum, something that is in
//! practice usually handled in ad-hoc ways. Here, the algebra of lens composition handles this case
//! naturally by using the get map, a seemingly trivial, unused piece of data for previous optimisers.
//! We finish off these examples by moving to a different base category POLYZ2 . This example shows
//! that our framework describes learning in not just continuous, but discrete settings too. Again, we
//! fix a parameterised map (Z𝑝 , 𝑓 ) : POLYZ2 (R𝑎 , R𝑏 ) but this time we fix the identity learning rate
//! (Example 3.10), instead of a constant one.
//! Example 4.4 (Basic learning in Boolean circuits). Fix XOR as the loss map (Example 3.5) and the
//! basic gradient update (Example 3.13). The put map again simplifies. The type reduces to 𝐴×𝑃 ×𝐵 → 𝑃
//! and the implementation to put(𝑎, 𝑝, 𝑏𝑡 ) = 𝑝 + 𝑝 ′ where (𝑝 ′, 𝑎 ′) = 𝑅 [𝑓 ] (𝑝, 𝑎, 𝑓 (𝑝, 𝑎) + 𝑏𝑡 ).
//! A sketch of learning iteration. Having described a number of examples in supervised learning,
//! we outline how to model learning iteration in our framework. Recall the aforementioned put map
//! whose type is 𝐴 × 𝑃 × 𝐵 → 𝑃 (for simplicity here modelled without state 𝑆). This map takes an
//! input-output pair (𝑎 0, 𝑏 0 ), the current parameter 𝑝𝑖 and produces an updated parameter 𝑝𝑖+1 . At the
//! next time step, it takes a potentially different input-output pair (𝑎 1, 𝑏 1 ), the updated parameter 𝑝𝑖+1
//! and produces 𝑝𝑖+2 . This process is then repeated. We can model this iteration as a composition of
//! the put map with itself, as a composite (𝐴 × put × 𝐵); put whose type is 𝐴 × 𝐴 × 𝑃 × 𝐵 × 𝐵 → 𝑃. This
//! map takes two input-output pairs 𝐴 × 𝐵, a parameter and produces a new parameter by processing
//! these datapoints in sequence. One can see how this process can be iterated any number of times,
//! and even represented as a string diagram.
//! But we note that with a slight reformulation of the put map, it is possible to obtain a conceptually
//! much simpler definition. The key insight lies in seeing that the map put : 𝐴 × 𝑃 × 𝐵 → 𝑃 is
//! essentially an endo-map 𝑃 → 𝑃 with some extra inputs 𝐴 × 𝐵; it’s a parameterised map!
//! In other words, we can recast the put map as a parameterised map (𝐴 × 𝐵, put) : Para(C) (𝑃, 𝑃).
//! Since it is an endo-map, it can be composed with itself. The resulting composite is too an endo-map,
//! which now takes two “parameters”: input-output pair at the time step 0 and time step 1. This
//! process can then be repeated, with Para composition automatically taking care of the algebra of
//! iteration.
//! 𝐴×𝐵
//!
//! 𝑃
//!
//! put
//!
//! 𝐴×𝐵
//! 𝑃
//!
//! put
//!
//! 𝐴×𝐵
//! . 𝑛. .
//!
//! put
//!
//! 𝑃
//!
//! This reformulation captures the essence of parameter iteration: one can think of it as a trajectory
//! 𝑝𝑖 , 𝑝𝑖+1, 𝑝𝑖+2, ... through the parameter space; but it is a trajectory parameterised by the dataset. With
//! different datasets the algorithm will take a different path through this space and learn different
//! things.
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! Categorical Foundations of Gradient-Based Learning
//!
//! 4.2
//!
//! 1:21
//!
//! Deep Dreaming: Supervised Learning of Inputs
//!
//! We have seen that attaching gradient descent to the parameter port of the parametric lens as a
//! reparameterisation allows us to learn the parameters in a supervised way. In this section we describe
//! how attaching the gradient descent lens to the input port provides us with a way to enhance an
//! input image to elicit a particular interpretation. This is the idea behind the technique called Deep
//! Dreaming, appearing in the literature in many forms [Dosovitskiy and Brox 2015; Mahendran and
//! Vedaldi 2014; Nguyen et al. 2014; Simonyan et al. 2014].
//! 𝑆 ×𝐴
//!
//! 𝑆 ×𝐴
//!
//! 𝑃
//!
//! 𝐵
//!
//! Optimiser
//!
//! 𝐴
//!
//! 𝐴′
//!
//! 𝐵′
//! 𝐴
//!
//! 𝐵
//!
//! 𝐴′
//!
//! 𝐿
//! Loss
//!
//! Model
//! 𝐵′
//!
//! 𝛼
//! 𝐿′
//!
//! Fig. 7. Deep dreaming: supervised learning of inputs
//!
//! Deep dreaming is a technique which uses the parameters 𝑝 of some trained classifier network
//! to iteratively dream up, or amplify some features of a class 𝑏 on a chosen input 𝑎. For example, if
//! we start with an image of a landscape 𝑎 0 , a label 𝑏 of a “cat” and a parameter 𝑝 of a sufficiently
//! well-trained classifier, we can start performing “learning” as usual: computing the predicted class
//! for the landscape 𝑎 0 for the network with parameters 𝑝, and then computing the distance between
//! the prediction and our label of a cat 𝑏. When performing backpropagation, the respective changes
//! computed for each layer tell us how the activations of that layer should have been changed to be
//! more “cat” like. This includes the first (input) layer of the landscape 𝑎 0 . Usually, we discard this
//! changes and apply gradient update to the parameters. In deep dreaming we discard the parameters
//! and apply gradient update to the input (Figure 7). Gradient update here takes these changes and
//! computes a new image 𝑎 1 which is the same image of the landscape, but changed slightly so to
//! look more like whatever the network thinks a cat looks like. This is the essence of deep dreaming,
//! where iteration of this process allows networks to dream up features and shapes on a particular
//! chosen image [Goo 2015].
//! Just like in the previous subsection, we can write this deep dreaming system as a map in
//! Para(Lens(C)) from (1, 1) to (1, 1) whose parameter space is (𝑆 × 𝐴 × 𝑃 × 𝐵, 𝑆 × 𝐴). In other words,
//! this is a lens of type (𝑆 × 𝐴 × 𝑃 × 𝐵, 𝑆 × 𝐴) → (1, 1) whose get map is trivial. Its put map has the
//! type 𝑆 × 𝐴 × 𝑃 × 𝐵 → 𝑆 × 𝐴 and unpacks to
//! put(𝑠, 𝑎, 𝑝, 𝑏𝑡 ) = 𝑈 ∗ (𝑠, 𝑎, 𝑎 ′)
//!
//! where
//!
//! 𝑎 = 𝑈 (𝑠, 𝑎)
//! 𝑏 𝑝 = 𝑓 (𝑝, 𝑎)
//! (𝑏𝑡′, 𝑏 𝑝′ ) = 𝑅 [loss] (𝑏𝑡 , 𝑏 𝑝 , 𝛼 (loss(𝑏𝑡 , 𝑏 𝑝 )))
//! (𝑝 ′, 𝑎 ′) = 𝑅 [𝑓 ] (𝑝, 𝑎, 𝑏 𝑝′ )
//!
//! We note that deep dreaming is usually presented without any loss function as a maximisation of
//! a particular activation in the last layer of the network output [Simonyan et al. 2014, Section 2.].
//! This maximisation is done with gradient ascent, as opposed to gradient descent. However, this is
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! 1:22
//!
//! G.S.H. Cruttwell, Bruno Gavranović, Neil Ghani, Paul Wilson, and Fabio Zanasi
//!
//! just a special case of our framework where the loss function is the dot product (Example 3.7). The
//! choice of the particular activation is encoded as a one-hot vector, and the loss function in that case
//! essentially masks the network output, leaving active only the particular chosen activation. The
//! final component is the gradient ascent: this is simply recovered by choosing a positive, instead of a
//! negative learning rate [Simonyan et al. 2014]. We explicitly unpack this in the following example.
//! Example 4.5 (Deep dreaming, dot product loss, gradient descent). Fix the base category to Smooth
//! and a parameterised map (R𝑝 , 𝑓 ) : Para(Smooth) (R𝑎 , R𝑏 ). Fix the dot product loss (Example 3.7),
//! basic gradient descent (Example 3.12), and a positive learning rate 𝛼 : R. Then the above put map
//! simplifies. Since there is no state, its type reduces to 𝐴 × 𝑃 × 𝐵 → 𝐴 and its implementation to
//! put(𝑎, 𝑝, 𝑏𝑡 ) = 𝑎 + 𝑎 ′
//!
//! where (𝑝 ′, 𝑎 ′) = 𝑅 [𝑓 ] (𝑝, 𝑎, 𝛼 · 𝑏𝑡 ).
//!
//! Like in Example 4.1, this update can be rewritten as
//! put(𝑎, 𝑝, 𝑏𝑡 ) = 𝑎 + 𝛼 · (𝑅 [𝑓 ] (𝑝, 𝑎, 𝑏𝑡 ); 𝜋1 )
//! making a few things apparent. This update does not depend on the prediction 𝑓 (𝑝, 𝑎): no matter
//! what the network has predicted, the goal is always to maximize particular activations. Which
//! activations? The ones chosen by 𝑏𝑡 . When 𝑏𝑡 is a one-hot vector, this picks out the activation of
//! just one class to maximize, which is often done in practice.
//! While we present only the most basic image, there is plenty of room left for exploration. The
//! work of [Simonyan et al. 2014, Section 2.] adds an extra regularization term to the image. In general,
//! the neural network 𝑓 is sometimes changed to copy a number of internal activations which are
//! then exposed on the output layer. Maximizing all these activations often produces more visually
//! appealing results. In the literature we did not find an example which uses the Softmax-cross entropy
//! (Example 3.6) as a loss function in deep dreaming, which seems like the more natural choice in this
//! setting. Furthermore, while deep dreaming is commonly done with basic gradient descent, there
//! is nothing preventing one from doing deep dreaming with any of the optimizer lenses discussed
//! in the previous section, or even doing deep dreaming in the context of Boolean circuits. Lastly,
//! learning iteration which was described in at the end of previous subsection can be modelled here
//! in an analogous way.
//! 5
//!
//! IMPLEMENTATION
//!
//! We provide a proof-of-concept implementation as a Python library.10 We demonstrate the correctness of our library empirically using a number of experiments implemented both in our library and
//! in Keras[Chollet et al. 2015], a popular framework for deep learning. For example, one experiment
//! is a model for the MNIST image classification problem[Lecun et al. 1998]: we implement the same
//! model in both frameworks and achieve comparable accuracy.
//! Our implementation also demonstrates the advantages of our approach. Firstly, computing the
//! gradients of the network is greatly simplified through the use of lens composition. Secondly, model
//! architectures can be expressed in a principled, mathematical language; as morphisms of a monoidal
//! category. Finally, the modularity of our approach makes it easy to see how various aspects of
//! training can be modified: for example, one can define a new optimization algorithm simply by
//! defining an appropriate lens. We now give a brief sketch of our implementation.
//! 10 Full
//!
//! usage examples, source code, and experiments using our proof-of-concept can be found at
//! https://github.com/statusfailed/numeric-optics-python/.
//!
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! Categorical Foundations of Gradient-Based Learning
//!
//! 5.1
//!
//! 1:23
//!
//! Constructing a Model with Lens and Para
//!
//! We model a lens (𝑓 , 𝑓 ∗ ) in our library with the Lens class, which consists of a pair of maps fwd
//! and rev corresponding to 𝑓 and 𝑓 ∗ , respectively. For example, we write the identity lens (1𝐴 , 𝜋2 )
//! as follows:
//! i d e n t i t y = Lens ( lambda x : x , lambda x_dy : x_dy [ 1 ] )
//!
//! The composition (in diagrammatic order) of Lens values f and g is written f » g, and monoidal
//! composition as f @ g. Similarly, the type of Para maps is modeled by the Para class, with composition and monoidal product written the same way. Our library provides several primitive Lens
//! and Para values.
//! Let us now see how to construct a single layer neural network from the composition of such
//! primitives. Diagramatically, we wish to construct the following model, representing a single ‘dense’
//! layer of a neural network:
//! R𝑏×𝑎 R𝑏×𝑎
//! R𝑎
//! R𝑎
//!
//! R𝑏 R𝑏
//!
//! R𝑏
//! linear
//!
//! R𝑏
//! activation
//!
//! bias
//! R𝑏
//!
//! R𝑏
//!
//! R𝑏
//! R𝑏
//!
//! (12)
//!
//! Here, the parameters of linear are the coefficients of a 𝑏 × 𝑎 matrix, and the underlying lens has
//! as its forward map the function (𝑀, 𝑥) → 𝑀 · 𝑥, where 𝑀 is the 𝑏 × 𝑎 matrix whose coefficients are
//! the R𝑏×𝑎 parameters, and 𝑥 ∈ R𝑎 is the input vector. The bias map is even simpler: the forward map
//! of the underlying lens is simply pointwise addition of inputs and parameters: (𝑏, 𝑥) → 𝑏 +𝑥. Finally,
//! the activation map simply applies a nonlinear function (e.g., sigmoid) to the input, and thus has
//! the trivial (unit) parameter space. The representation of this composition in code is straightforward:
//! we can simply compose the three primitive Para maps as in (12):
//! def d e n s e ( a , b , a c t i v a t i o n ) :
//! return l i n e a r ( a , b ) >> b i a s ( b ) >> a c t i v a t i o n
//!
//! Note that by constructing model architectures in this way, the computation of reverse derivatives
//! is greatly simplified: we obtain the reverse derivative ‘for free’ as the put map of the model.
//! Furthermore, adding new primitives is also simplified: the user need simply provide a function
//! and its reverse derivative in the form of a Para map. Finally, notice also that our approach is
//! truly compositional: we can define a hidden layer neural network with 𝑛 hidden units simply by
//! composing two dense layers, as follows:
//! d e n s e ( a , n , a c t i v a t i o n ) >> d e n s e ( n , b , a c t i v a t i o n )
//!
//! 5.2
//!
//! Learning
//!
//! Now that we have constructed a model, we also need to use it to learn from data. Concretely, we will
//! construct a full parametric lens as in Figure 6 then extract its put map to iterate over the dataset.
//! By way of example, let us see how to construct the following parametric lens, representing basic
//! gradient descent over a single layer neural network with a fixed learning rate:
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! 1:24
//!
//! G.S.H. Cruttwell, Bruno Gavranović, Neil Ghani, Paul Wilson, and Fabio Zanasi
//!
//! 𝐴
//!
//! 𝑃 𝑃
//!
//! 𝐵
//!
//! +
//! 𝑃
//!
//! 𝑃′
//!
//! 𝐴
//!
//! 𝐵′
//! 𝐵
//!
//! dense
//! 𝐴′
//!
//! 𝐿
//! loss
//!
//! 𝐵′
//!
//! (13)
//! 𝐿′
//!
//! 𝜖
//!
//! This morphism is constructed essentially as below, where apply_update(𝛼, 𝑓 ) represents the
//! ‘vertical stacking’ of 𝛼 atop 𝑓 :
//! a p p l y _ u p d a t e ( b a s i c _ u p d a t e , d e n s e ) >> l o s s >> l e a r n i n g _ r a t e ( 𝜖 )
//!
//! Now, given the parametric lens of (13), one can construct a morphism step : 𝐵 × 𝑃 × 𝐴 → 𝑃
//! which is simply the put map of the lens. Training the model then consists of iterating the step
//! function over dataset examples (𝑥, 𝑦) ∈ 𝐴 × 𝐵 to optimise some initial choice of parameters 𝜃 0 ∈ 𝑃,
//! by letting 𝜃 𝑖+1 = step(𝑦𝑖 , 𝜃 𝑖 , 𝑥𝑖 ).
//! Note that our library also provides a utility function to construct step from its various pieces:
//! s t e p = s u p e r v i s e d _ s t e p ( model , u p d a t e , l o s s , l e a r n i n g _ r a t e )
//!
//! For an end-to-end example of model training and iteration, we refer the interested reader to the
//! experiments accompanying the code: https://github.com/statusfailed/numeric-optics-python/.
//! 6
//!
//! RELATED WORK
//!
//! The work [Fong et al. 2017] is closely related to ours, in that it provides an abstract categorical
//! model of back-propagation. However, it differs in a number of key aspects. We give a complete
//! lens-theoretic explanation of what is back-propagated via (i) the use of CRDCs to model gradients;
//! and (ii) the Para construction to model parameterized functions and parameter update. We thus
//! can go well beyond [Fong et al. 2017] in terms of examples - their example of smooth functions
//! and basic gradient descent is covered in our subsection 4.1.
//! We also explain some of the constructions of [Fong et al. 2017] in a more structured way. For
//! example, rather than considering the category Learn of [Fong et al. 2017] as primitive, here we
//! construct it as a composite of two more basic constructions (the Para and Lens constructions).
//! The flexibility could be used, for example, to compositionally replace Para with a variant allowing
//! parameters to come from a different category, or lenses with the category of optics [Riley 2018]
//! enabling us to model things such as control flow using prisms.
//! One more important thing is related to functoriality. We use a functor to augment a parameterised
//! map with its backward pass, just like [Fong et al. 2017]. However, they additionally augmented
//! this map with a loss map and gradient descent using a functor as well. This added extra conditions
//! on the partial derivatives of the loss function: it needed to be invertible in the 2nd variable. This
//! constraint was not justified in [Fong et al. 2017], nor is it a constraint that appears in machine
//! learning practice. This led us to reexamine their constructions, coming up with our reformulation
//! that does not require it. While loss maps and optimisers are mentioned in [Fong et al. 2017] as
//! parts of the aforementioned functor, here they are extracted out and play a key role: loss maps
//! are parameterised lenses and optimisers are reparameterisations. Thus, in this paper we instead
//! use Para-composition to add the loss map to the model, and Para 2-cells to add optimisers. The
//! mentioned inverse of the partial derivative of the loss map in the 2nd variable was also hypothesised
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! Categorical Foundations of Gradient-Based Learning
//!
//! 1:25
//!
//! to be relevant to deep dreaming. In our paper we have given a complete picture of deep dreaming
//! systems, showing it is gradient update which is used to dream up pictures.
//! We also correct a small issue in Theorem III.2 of [Fong et al. 2017]. There, the morphisms of Learn
//! were defined up to an equivalence (pg. 4 of [Fong et al. 2017]) but, unfortunately, the functor defined
//! in Theorem III.2 does not respect this equivalence relation. Our approach instead uses 2-cells which
//! comes from the universal property of Para — a 2-cell from (𝑃, 𝑓 ) : 𝐴 → 𝐵 to (𝑄, 𝑔) : 𝐴 → 𝐵 is a
//! lens, and hence has two components: a map 𝛼 : 𝑄 → 𝑃 and 𝛼 ∗ : 𝑄 × 𝑃 → 𝑄. By comparison, we
//! can see the equivalence relation of [Fong et al. 2017] as being induced by map 𝛼 : 𝑄 → 𝑃, and not a
//! lens. Our approach highlights the importance of the 2-categorical structure of learners. In addition,
//! it does not treat the functor Para(C) → Learn as a primitive. In our case, this functor has the type
//! Para(C) → Para(Lens(C)) and arises from applying Para to a canonical functor C → Lens(C)
//! existing for any reverse derivative category, not just Smooth. Lastly, in our paper we have taken
//! the advantage of the graphical calculus for Para, redrawing many diagrams appearing in [Fong
//! et al. 2017] in a structured way.
//! Other than [Fong et al. 2017], there are a few more relevant papers. The work of ([Dalrymple
//! 2019]) contains a sketch of some of the ideas this paper evolved from. They are based on the
//! interplay of optics with parameterisation, albeit framed in the setting diffeological spaces, and
//! requiring cartesian and local cartesian closed structure on the base category. Lenses and Learners
//! are studied in the eponymous work of [Fong and Johnson 2019] which observes that lenses are
//! parameterised learners. They do not explore any of the relevant Para or CRDC structure, but make
//! the distinction between symmetric and asymmetric lenses, studying how they are related to learners
//! defined in [Fong et al. 2017]. A lens-like implementation of automatic differentiation is the focus of
//! [Elliott 2018], but learning algorithms aren’t studied. A relationship between category-theoretic
//! perspective on probabilistic modeling and gradient-based optimisation is studied in [Shiebler 2020]
//! which also studies a variant of the Para construction. Usage of Cartesian differential categories to
//! study learning is found in [Sprunger and Katsumata 2019]. They extend the differential operator to
//! work on stateful maps, but do not study lenses, parameterisation nor update maps. The work of
//! [Gavranovic 2019] studies deep learning in the context of Cycle-consistent Generative Adversarial
//! Networks [Zhu et al. 2017] and formalises it via free and quotient categories, making parallels to
//! the categorical formulations of database theory [Spivak 2010]. They do use the Para construction,
//! but do not relate it to lenses nor reverse derivative categories.
//! Lastly, the concept of parameterised lenses has started appearing in recent formulations of
//! categorical game theory and cybernetics [Capucci et al. 2021; Capucci et al. 2021]. The work of
//! [Capucci et al. 2021] generalises the study of parameterised lenses into parameterised optics and
//! connects it to game thereotic concepts such as Nash equilibria. A general survey of category
//! theoretic approaches to machine learning, covering many of the above papers in detail, can be
//! found in [Shiebler et al. 2021].
//! 7
//!
//! CONCLUSIONS AND FUTURE DIRECTIONS
//!
//! We have given a categorical foundation of gradient-based learning algorithms which achieves a
//! number of important goals. The foundation is principled and mathematically clean, based on the
//! fundamental idea of a parameterised lens. The foundation covers a wide variety of examples: it
//! covers different optimisers and loss maps in gradient-based learning, it covers different settings
//! where gradient-based learning happens (smooth functions vs. boolean circuits) and it covers both
//! learning of parameters and learning of inputs (deep dreaming). Finally, the foundation is more than
//! a mere abstraction: we have also shown how it can be used to give a practical implementation of
//! learning, as discussed in section 5.
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! 1:26
//!
//! G.S.H. Cruttwell, Bruno Gavranović, Neil Ghani, Paul Wilson, and Fabio Zanasi
//!
//! There are a number of important directions which are possible to explore because of this work.
//! One of the most exciting ones is the extension to more complex neural network architectures.
//! Our formulation of the loss map as a parameterised lens should pave the way for Generative
//! Adversarial Networks [Goodfellow et al. 2014], an exciting new architecture whose loss map can be
//! said to be learned in tandem with the base network. In all our settings we have fixed an optimiser
//! beforehand. The work of [Andrychowicz et al. 2016] describes a meta-learning approach which sees
//! the optimiser as a neural network whose parameters and gradient update rule can be learned. This
//! is an exciting prospect since one can model optimisers as parameterised lenses; and our framework
//! covers learning with parameterised lenses. Recurrent neural networks are another example of a
//! more complex architecture, which has already been studied in the context of differential categories
//! in [Sprunger and Katsumata 2019]. When it comes to architectures, future work includes modelling
//! some classical systems as well, such as the Support Vector Machines [Cortes and Vapnik 1995],
//! which should be possible with the usage of loss maps such as Hinge loss.
//! We have not made use of the full power of the CRDC axioms; in particular, we did not explicitly
//! need axioms RD.6 or RD.7, which deal with the behaviour of higher-order derivatives. However,
//! some supervised learning algorithms do use the higher-order derivatives (for example, the Hessian)
//! for additional optimisations; as such, future work includes exploring how to use those axioms to
//! capture these optimisations. Taking this idea in a different direction, one can see that much of our
//! work can be applied to any functor of the form 𝐹 : C → Lens(C) - 𝐹 does not necessarily have to
//! be of the form 𝑓 ↦→ (𝑓 , 𝑅 [𝑓 ]) for a CRDC 𝑅. Moreover, by working with more generalised forms of
//! the lens category (such as dependent lenses), we may be able to capture ideas related to supervised
//! learning on manifolds. And, of course, we can vary the parameter space to endow it with different
//! structure from the functions we wish to learn. In this vein, we wish to use fibrations/dependent types
//! to model the use of tangent bundles: this would foster the extension of the correct by construction
//! paradigm to machine learning, and thereby addressing the widely acknowledged problem of trusted
//! machine learning. The possibilities are made much easier by the compositional nature of our
//! framework. Another key topic for future work is to link gradient-based learning with game theory.
//! At a high level, the former takes little incremental steps to achieve an equilibrium while the later
//! aims to do so in one fell swoop. Formalising this intuition is possible with our lens-based framework
//! and the lens-based framework for game theory [Ghani et al. 2016]. Finally, because our framework is
//! quite general, in future work we plan to consider further modifications and additions to encompass
//! non-supervised, probabilistic and non-gradient based learning. This includes genetic algorithms
//! and reinforcement learning.
//! REFERENCES
//! 2015. Inceptionism: Going Deeper into Neural Networks. (2015). https://ai.googleblog.com/2015/06/inceptionism-goingdeeper-into-neural.html
//! 2019. Explainable AI: the Basics - Policy Briefing. royalsociety.org/ai-interpretability
//! S. Abramsky and B. Coecke. 2004. A categorical semantics of quantum protocols. In Proceedings of the 19th Annual IEEE
//! Symposium on Logic in Computer Science, 2004. 415–425. https://doi.org/10.1109/LICS.2004.1319636
//! Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W. Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and
//! Nando de Freitas. 2016. Learning to learn by gradient descent by gradient descent. arXiv e-prints, Article arXiv:1606.04474
//! (June 2016), arXiv:1606.04474 pages. arXiv:1606.04474 [cs.NE]
//! John C. Baez and Jason Erbele. 2014. Categories in Control. arXiv e-prints, Article arXiv:1405.6881 (May 2014),
//! arXiv:1405.6881 pages. arXiv:1405.6881 [math.CT]
//! R. Blute, R. Cockett, and R. Seely. 2009. Cartesian Differential Categories. Theory and Applications of Categories 22 (2009),
//! 622–672.
//! Aaron Bohannon, J. Nathan Foster, Benjamin C. Pierce, Alexandre Pilkiewicz, and Alan Schmitt. 2008. Boomerang:
//! Resourceful Lenses for String Data. SIGPLAN Not. 43, 1 (Jan. 2008), 407–419. https://doi.org/10.1145/1328897.1328487
//!
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! Categorical Foundations of Gradient-Based Learning
//!
//! 1:27
//!
//! Guillaume Boisseau. 2020.
//! String Diagrams for Optics.
//! arXiv e-prints, Article arXiv:2002.11480 (Feb. 2020),
//! arXiv:2002.11480 pages. arXiv:2002.11480 [math.CT]
//! Filippo Bonchi, Pawel Sobocinski, and Fabio Zanasi. 2017. The Calculus of Signal Flow Diagrams I: Linear relations on
//! streams. Inf. Comput. 252 (2017), 2–29. https://doi.org/10.1016/j.ic.2016.03.002
//! Matteo Capucci, Bruno Gavranovi’c, Jules Hedges, and E. F. Rischel. 2021. Towards foundations of categorical cybernetics.
//! Matteo Capucci, Neil Ghani, Jérémy Ledent, and Fredrik Nordvall Forsberg. 2021. Translating Extensive Form Games to Open
//! Games with Agency. arXiv e-prints, Article arXiv:2105.06763 (May 2021), arXiv:2105.06763 pages. arXiv:2105.06763 [cs.GT]
//! Francois Chollet et al. 2015. Keras. https://github.com/fchollet/keras
//! Bryce Clarke, Derek Elkins, Jeremy Gibbons, Fosco Loregian, Bartosz Milewski, Emily Pillmore, and Mario Román. 2020.
//! Profunctor optics, a categorical update. arXiv e-prints, Article arXiv:2001.07488 (Jan. 2020), arXiv:2001.07488 pages.
//! arXiv:2001.07488 [cs.PL]
//! J. Robin B. Cockett, Geoff S. H. Cruttwell, Jonathan Gallagher, Jean-Simon Pacaud Lemay, Benjamin MacAdam, Gordon D.
//! Plotkin, and Dorette Pronk. 2019. Reverse derivative categories. In CSL.
//! Bob Coecke and Aleks Kissinger. 2017. Picturing Quantum Processes: A First Course in Quantum Theory and Diagrammatic
//! Reasoning. Cambridge University Press. https://doi.org/10.1017/9781316219317
//! Corinna Cortes and Vladimir Vapnik. 1995. Support-vector networks. Machine learning 20, 3 (1995), 273–297.
//! Matthieu Courbariaux, Yoshua Bengio, and Jean-Pierre David. [n.d.]. BinaryConnect: Training Deep Neural Networks with
//! binary weights during propagations. arXiv:1511.00363 [cs] ([n. d.]).
//! G. S. H. Cruttwell. 2012. Cartesian differential categories revisited. arXiv e-prints, Article arXiv:1208.4070 (Aug. 2012),
//! arXiv:1208.4070 pages. arXiv:1208.4070 [math.CT]
//! David Dalrymple. 2019. Dioptics: a Common Generalization of Open Games and Gradient-Based Learners. SYCO7
//! (2019). https://research.protocol.ai/publications/dioptics-a-common-generalization-of-open-games-and-gradient-basedlearners/dalrymple2019.pdf
//! Alexey Dosovitskiy and Thomas Brox. 2015. Inverting Convolutional Networks with Convolutional Networks. CoRR
//! abs/1506.02753 (2015). arXiv:1506.02753 http://arxiv.org/abs/1506.02753
//! John Duchi, Elad Hazan, and Yoram Singer. 2011. Adaptive subgradient methods for online learning and stochastic
//! optimization. Journal of Machine Learning Research 12, Jul (2011), 2121–2159.
//! Conal Elliott. 2018. The simple essence of automatic differentiation (Differentiable functional programming made easy).
//! CoRR abs/1804.00746 (2018). arXiv:1804.00746 http://arxiv.org/abs/1804.00746
//! Brendan Fong and Michael Johnson. 2019. Lenses and Learners. CoRR abs/1903.03671 (2019). arXiv:1903.03671 http:
//! //arxiv.org/abs/1903.03671
//! Brendan Fong, David I. Spivak, and Rémy Tuyéras. 2017. Backprop as Functor: A compositional perspective on supervised
//! learning. CoRR abs/1711.10455 (2017). arXiv:1711.10455 http://arxiv.org/abs/1711.10455
//! Bruno Gavranovic. 2019. Compositional Deep Learning. ArXiv abs/1907.08292 (2019).
//! Neil Ghani, Jules Hedges, Viktor Winschel, and Philipp Zahn. 2016. Compositional game theory. arXiv e-prints, Article
//! arXiv:1603.04641 (March 2016), arXiv:1603.04641 pages. arXiv:1603.04641 [cs.GT]
//! Dan R. Ghica, Achim Jung, and Aliaume Lopez. 2017. Diagrammatic Semantics for Digital Circuits. arXiv e-prints, Article
//! arXiv:1703.10247 (March 2017), arXiv:1703.10247 pages. arXiv:1703.10247 [cs.PL]
//! Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua
//! Bengio. 2014. Generative Adversarial Nets. In Advances in Neural Information Processing Systems 27, Z. Ghahramani,
//! M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger (Eds.). Curran Associates, Inc., 2672–2680. http://papers.
//! nips.cc/paper/5423-generative-adversarial-nets.pdf
//! Andreas Griewank and Andrea Walther. 2008. Evaluating derivatives: principles and techniques of algorithmic differentiation.
//! Society for Industrial and Applied Mathematics.
//! Jules Hedges. 2018. Limits of bimorphic lenses. arXiv e-prints, Article arXiv:1808.05545 (Aug. 2018), arXiv:1808.05545 pages.
//! arXiv:1808.05545 [math.CT]
//! Michael Johnson, Robert Rosebrugh, and R.J. Wood. 2012. Lenses, fibrations and universal translations. Mathematical
//! structures in computer science 22 (2012), 25–42.
//! Diederik P. Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. In 3rd International Conference on
//! Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, Yoshua Bengio and
//! Yann LeCun (Eds.). http://arxiv.org/abs/1412.6980
//! Yann Lecun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. 1998. Gradient-Based Learning Applied to Document
//! Recognition. In Proceedings of the IEEE. 2278–2324. https://doi.org/10.1109/5.726791
//! Aravindh Mahendran and Andrea Vedaldi. 2014. Understanding Deep Image Representations by Inverting Them. CoRR
//! abs/1412.0035 (2014). arXiv:1412.0035 http://arxiv.org/abs/1412.0035
//! Anh Mai Nguyen, Jason Yosinski, and Jeff Clune. 2014. Deep Neural Networks are Easily Fooled: High Confidence Predictions
//! for Unrecognizable Images. CoRR abs/1412.1897 (2014). arXiv:1412.1897 http://arxiv.org/abs/1412.1897
//!
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! 1:28
//!
//! G.S.H. Cruttwell, Bruno Gavranović, Neil Ghani, Paul Wilson, and Fabio Zanasi
//!
//! Christopher Olah. 2015. Neural Networks, Types, and Functional Programming. (2015). http://colah.github.io/posts/201509-NN-Types-FP/
//! B.T. Polyak. 1964. Some methods of speeding up the convergence of iteration methods. U. S. S. R. Comput. Math. and Math.
//! Phys. 4, 5 (1964), 1 – 17. https://doi.org/10.1016/0041-5553(64)90137-5
//! Mitchell Riley. 2018. Categories of Optics. arXiv:1809.00738 [math.CT]
//! Peter Selinger. 2001. Control categories and duality: on the categorical semantics of the lambda-mu calculus. Mathematical
//! Structures in Computer Science 11, 02 (4 2001), 207–260. https://doi.org/null
//! P. Selinger. 2010. A Survey of Graphical Languages for Monoidal Categories. Lecture Notes in Physics (2010), 289–355.
//! https://doi.org/10.1007/978-3-642-12821-9_4
//! Sanjit A. Seshia and Dorsa Sadigh. 2016. Towards Verified Artificial Intelligence. CoRR abs/1606.08514 (2016). arXiv:1606.08514
//! http://arxiv.org/abs/1606.08514
//! Dan Shiebler. 2020. Categorical Stochastic Processes and Likelihood. arXiv e-prints, Article arXiv:2005.04735 (May 2020),
//! arXiv:2005.04735 pages. arXiv:2005.04735 [cs.AI]
//! Dan Shiebler, Bruno Gavranović, and Paul Wilson. 2021. Category Theory in Machine Learning. arXiv e-prints, Article
//! arXiv:2106.07032 (June 2021), arXiv:2106.07032 pages. arXiv:2106.07032 [cs.LG]
//! Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. 2014. Deep Inside Convolutional Networks: Visualising Image
//! Classification Models and Saliency Maps. arXiv:1312.6034 [cs.CV]
//! David I. Spivak. 2010. Functorial Data Migration. CoRR abs/1009.1166 (2010). arXiv:1009.1166 http://arxiv.org/abs/1009.1166
//! David Sprunger and Shin-ya Katsumata. 2019. Differentiable Causal Computations via Delayed Trace. CoRR abs/1903.01093
//! (2019). arXiv:1903.01093 http://arxiv.org/abs/1903.01093
//! Albert Steckermeier. 2015. Lenses in Functional Programming. Preprint (2015).
//! Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. 2013. On the importance of initialization and momentum
//! in deep learning. In Proceedings of the 30th International Conference on Machine Learning (Proceedings of Machine
//! Learning Research, Vol. 28), Sanjoy Dasgupta and David McAllester (Eds.). PMLR, Atlanta, Georgia, USA, 1139–1147.
//! http://proceedings.mlr.press/v28/sutskever13.html
//! D. Turi and G. Plotkin. 1997. Towards a mathematical operational semantics. In Proceedings of Twelfth Annual IEEE
//! Symposium on Logic in Computer Science. 280–291. https://doi.org/10.1109/LICS.1997.614955
//! Paul Wilson and Fabio Zanasi. 2020. Reverse Derivative Ascent: A Categorical Approach to Learning Boolean Circuits.
//! EPTCS (2020). https://cgi.cse.unsw.edu.au/~eptcs/paper.cgi?ACT2020:31
//! Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. 2017. Unpaired Image-to-Image Translation using CycleConsistent Adversarial Networks. arXiv e-prints, Article arXiv:1703.10593 (March 2017), arXiv:1703.10593 pages.
//! arXiv:1703.10593 [cs.CV]
//!
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! Categorical Foundations of Gradient-Based Learning
//!
//! A
//!
//! 1:29
//!
//! BACKGROUND ON CARTESIAN REVERSE DIFFERENTIAL CATEGORIES
//!
//! Here we briefly review the definitions of Cartesian left additive category (CLAC), Cartesian reverse
//! differential category (CRDC) and additive and linear maps in these categories. Note that in this
//! appendix we follow the convention of [Cockett et al. 2019] and write composition in diagrammatic
//! order by juxtaposition of terms (rather than a semicolon) to shorten the form of many of the
//! expressions.
//! Definition A.1. A category C is said to be Cartesian when there are chosen binary products ×,
//! with projection maps 𝜋𝑖 and pairing operation ⟨−, −⟩, and a chosen terminal object 𝑇 , with unique
//! maps ! to the terminal object.
//! Definition A.2. A left additive category [Blute et al. 2009, Definition 1.1.1] (CLAC) is a category
//! C such that each hom-set has commutative monoid structure, with addition operation + and zero
//! maps 0, such that composition on the left preserves the additive structure: for any appropriate
//! 𝑓 , 𝑔, ℎ, 𝑓 (𝑔 + ℎ) = 𝑓 𝑔 + 𝑓 ℎ and 𝑓 0 = 0.
//! Definition A.3. A map ℎ : 𝑋 → 𝑌 in a CLAC is additive if it has the property that it preserves
//! additive structure by composition on the right: for any maps 𝑥, 𝑦 : 𝑍 → 𝑋 , (𝑥 + 𝑦); ℎ = 𝑥; ℎ + 𝑦; ℎ,
//! and 0; ℎ = 0.
//! Definition A.4. A Cartesian left additive category [Blute et al. 2009, Definition 1.2.1] is a left
//! additive category C which is Cartesian and such that all projection maps 𝜋𝑖 are additive.
//! The central definition of [Cockett et al. 2019] is the following:
//! Definition A.5. A Cartesian reverse differential category (CRDC) is a Cartesian left additive
//! category C which has, for each map 𝑓 : 𝐴 → 𝐵 in C, a map
//! 𝑅 [𝑓 ] : 𝐴 × 𝐵 → 𝐴
//! satisfying seven axioms:
//! [RD.1] 𝑅 [𝑓 + 𝑔] = 𝑅 [𝑓 ] + 𝑅 [𝑔] and 𝑅 [0] = 0.
//! [RD.2] ⟨𝑎, 𝑏 + 𝑐⟩𝑅 [𝑓 ] = ⟨𝑎, 𝑏⟩𝑅 [𝑓 ] + ⟨𝑎, 𝑐⟩𝑅 [𝑓 ] and ⟨𝑎, 0⟩𝑅 [𝑓 ] = 0.
//! [RD.3] 𝑅 [1] = 𝜋1 , 𝑅 [𝜋 0 ] = 𝜋 1𝜄 0 , and 𝑅 [𝜋1 ] = 𝜋1𝜄 1 . [RD.4] For a tupling of maps 𝑓 and 𝑔, the
//! following equality holds:
//! 𝑅 [⟨𝑓 , 𝑔⟩] = (1 × 𝜋 0 ); 𝑅 [𝑓 ] + (1 × 𝜋1 ); 𝑅 [𝑔]
//! And if !𝐴 : 𝐴 → 𝑇 is the unique map to the terminal object, 𝑅 [!𝐴 ] = 0.
//! [RD.5] For composable maps 𝑓 and 𝑔,
//! 𝑅 [𝑓 𝑔] = ⟨𝜋0, 𝜋0 𝑓 , 𝜋1 ⟩⟩(1 × 𝑅 [𝑔])𝑅 [𝑓 ]
//! [RD.6]
//! ⟨1 × 𝜋0, 0 × 𝜋1 ⟩(𝜄 0 × 1)𝑅 [𝑅 [𝑅 [𝑓 ]]]𝜋1 = (1 × 𝜋 1 )𝑅 [𝑓 ].
//! [RD.7]
//! (𝜄 0 × 1); 𝑅 [𝑅 [(𝜄 0 × 1)𝑅 [𝑅 [𝑓 ]]𝜋 1 ]]; 𝜋1 = ex; (𝜄 0 × 1)𝑅 [𝑅 [(𝜄 0 × 1)𝑅 [𝑅 [𝑓 ]]𝜋1 ]]𝜋1
//! (where ex is the map that exchanges the middle two variables).
//! As discussed in [Cockett et al. 2019], these axioms correspond to familiar properties of the reverse
//! derivative:
//! • [RD.1] says that differentiation preserves addition of maps, while [RD.2] says that differentiation is additive in its vector variable.
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//! 1:30
//!
//! G.S.H. Cruttwell, Bruno Gavranović, Neil Ghani, Paul Wilson, and Fabio Zanasi
//!
//! • [RD.3] and [RD.4] handle the derivatives of identities, projections, and tuples.
//! • [RD.5] is the (reverse) chain rule.
//! • [RD.6] says that the reverse derivative is linear in its vector variable.
//! • [RD.7] expresses the independence of order of mixed partial derivatives.
//!
//! Proc. ACM Program. Lang., Vol. 1, No. CONF, Article 1. Publication date: January 2018.
//!
//!
//! --- PAGE BREAK ---
//!
//! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//!
