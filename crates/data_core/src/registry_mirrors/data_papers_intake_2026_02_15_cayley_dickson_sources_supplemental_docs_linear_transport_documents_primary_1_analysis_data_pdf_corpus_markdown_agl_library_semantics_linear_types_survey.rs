//! # Extracted text: linear_types_survey.pdf
//!
//! - source_root: `/home/eirikr/Documents/AGL_Library/Semantics_Documentation`
//! - source_relpath: `linear_types_survey.pdf`
//! - source_abs: `/home/eirikr/Documents/AGL_Library/Semantics_Documentation/linear_types_survey.pdf`
//! - detected_kind: `pdf`
//! - extracted_at_utc: `2026-01-02T17:31:19+00:00`
//! - pages: `9`
//! - title: ``
//! - author: ``
//! - subject: ``
//! - keywords: ``
//! - creation_date: `Mon Nov  9 18:05:27 2020 PST`
//! - mod_date: `Mon Nov  9 18:05:27 2020 PST`
//! - encrypted: `no`
//!
//! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~text
//! Two-Stream Appearance Transfer Network for Person Image Generation
//! Chengkang Shen 1 ∗ , Peiyan Wang 2 * ,Wei Tang 1
//! 1
//!
//! arXiv:2011.04181v1 [cs.CV] 9 Nov 2020
//!
//! University of Illinois at Chicago
//! 2
//! Purdue University
//! {cshen26,tangw}@uic.edu, wang5035@purdue.edu
//!
//! Abstract
//! Pose guided person image generation means to generate a
//! photo-realistic person image conditioned on an input person
//! image and a desired pose. This task requires spatial manipulation of the source image according to the target pose. However, the generative adversarial networks (GANs) widely used
//! for image generation and translation rely on spatially local
//! and translation equivariant operators, i.e., convolution, pooling and unpooling, which cannot handle large image deformation. This paper introduces a novel two-stream appearance transfer network (2s-ATN) to address this challenge.
//! It is a multi-stage architecture consisting of a source stream
//! and a target stream. Each stage features an appearance transfer module and several two-stream feature fusion modules.
//! The former finds the dense correspondence between the twostream feature maps and then transfers the appearance information from the source stream to the target stream. The latter exchange local information between the two streams and
//! supplement the non-local appearance transfer. Both quantitative and qualitative results indicate the proposed 2s-ATN
//! can effectively handle large spatial deformation and occlusion
//! while retaining the appearance details. It outperforms prior
//! states of the art on two widely used benchmarks.
//!
//! 1
//!
//! Introduction
//!
//! Pose guided person image generation aims to transform a
//! person image from a source pose to a target pose while retaining the appearance details. It serves as a fundamental
//! tool for several practical applications such as image editing, video generation and data augmentation for person reidentification and action recognition (Yang et al. 2018; Zhu
//! et al. 2019; Qian et al. 2018). This task is very challenging especially in case of large pose transform, occlusion and
//! complex texture.
//! Convolutional neural networks (CNNs) (LeCun et al.
//! 1998) and their variants (Ronneberger, Fischer, and Brox
//! 2015; He et al. 2016), trained in an adversarial fashion
//! (Goodfellow et al. 2014), have been widely used for image generation and translation (Isola et al. 2017; Ledig et al.
//! 2017; Johnson, Alahi, and Fei-Fei 2016; Zhu et al. 2017;
//! Mirza and Osindero 2014). However, since CNNs are composed of spatially local and translation equivariant operators,
//! * The authors have contributed equally to this work.
//! Copyright © 2021, Association for the Advancement of Artificial
//! Intelligence (www.aaai.org). All rights reserved.
//!
//! Figure 1: Transferring the pose of a person requires spatial
//! manipulation of the source image.
//!
//! i.e., convolution, pooling and unpooling, they do not have an
//! explicit mechanism to handle articulated body deformation,
//! as illustrated in Fig. 1. To resolve this difficult issue, two
//! strategies have been adopted in prior person image generation approaches, i.e., parametric geometric transformation
//! and nonparametric dense flow. For example, Siarohin et al.
//! (Siarohin et al. 2018) apply an affine transformation to the
//! features of each body part region to deal with pixel-to-pixel
//! misalignment caused by the pose difference. However, it
//! cannot handle occlusion or out-of-plane rotation well. Some
//! methods (Han et al. 2019; Liu et al. 2019) predict the dense
//! flow field between the source and target images and apply
//! it to warp the feature maps. However, since the flow is predicted via a CNN, it cannot account for large or non-local
//! motion.
//! This paper introduces a novel two-stream appearance
//! transfer network (2s-ATN) for person image generation. It
//! can effectively handle large pose transform and occlusion
//! while retaining the appearance details. As illustrated in Fig.
//! 2, the 2s-ATN is a multi-stage architecture consisting of a
//! source stream and a target stream. The two streams respectively take as input the source pose and image and the target pose. At each stage is an appearance transfer block (ATblock). It consists of a novel appearance transfer module
//! (AT-module) and several two-stream feature fusion modules.
//! The AT-module finds the dense correspondence between the
//! two-stream feature maps and then transfers the appearance
//! information from the source stream to the target stream. Unlike the parametric geometric transformation or the nonparametric flow field, our AT-module is inspired by the selfattention (Vaswani et al. 2017) and performs a query-and-
//!
//!
//! --- PAGE BREAK ---
//! Figure 2: An overview of the proposed two-stream appearance transfer network (2s-ATN). It is a multi-stage architecture
//! consisting of two streams. The target and source streams respectively take as input the target pose and the source pose and
//! image, and pass them through convolutional encoders. Each stage is an appearance transfer block (AT-block). It consists of an
//! appearance transfer module (detailed in Fig. 3) and two two-stream feature fusion modules. We consider two fusion functions,
//! i.e., concatenation and summation, and find the former works better for both fusion modules. The target feature map from the
//! last AT-block passes through a convolutional decoder to generate a new person image with the same appearance as the source
//! image but in the target pose.
//! transfer procedure. But different with the self-attention, the
//! queries, keys and values in our AT-module have explicit semantic meaning, and they are specially designed for poseguided appearance transfer. Specifically, the feature vector
//! at each spatial location in the target stream is taken as a
//! query to match the key feature vectors in the source stream
//! so that the corresponding appearance values in the source
//! stream can be transferred to the desired location in the target
//! stream. Meanwhile, the two-stream feature fusion modules
//! allow local information exchange between the two streams
//! to supplement the non-local appearance transfer. Finally, the
//! target stream outputs a generated image with the source appearance but in the target pose.
//! Ablation study indicates that the proposed 2s-ATN can
//! effectively handle large spatial deformation and occlusion,
//! and improve the quality of the generated person images. Experimental results on two benchmark datasets, i.e., Market1501 (Zheng et al. 2015) and DeepFashion (Liu et al. 2016),
//! demonstrate the proposed approach outperforms state-ofthe-art methods both qualitatively and quantitatively.
//! The main technical contribution of this paper is the 2sATN and the AT-module. The 2s-ATN is a novel two-stream
//! and multi-stage network architecture for person image generation. It progressively transfers the appearance from the
//! source stream to the target stream guided by their spatial correspondence. Each stage combines a non-local AT-module
//! and several local two-stream feature fusion modules. The
//!
//! proposed AT-module is the first of its kind to use the target stream to query and transfer the source stream. The local and non-local modules are complementary, and they together enable the network to effectively handle large pose
//! deformation and occlusion.
//!
//! 2
//! 2.1
//!
//! Related Work
//!
//! Person Image Generation
//!
//! The task of pose guided person image generation was first
//! introduced by (Ma et al. 2017). Their two-stage network
//! first generates a coarse target image and then refines it in
//! an adversarial way. (Ma et al. 2018) disentangles the foreground, background and pose information, and then manipulates them to get the desired pose. The controllability of
//! the generation process is improved, but the quality of the
//! generated image is reduced. (Esser, Sutter, and Ommer
//! 2018) combines VAE (Kingma and Welling 2013) and UNet (Isola et al. 2017) to distinguish the appearance and
//! pose of a person image. However, it is difficult to represent the appearance features as a low-dimension underlying
//! code, which unavoidably loses information. (Siarohin et al.
//! 2018) introduces deformable skip connections to transform
//! the texture spatially. It uses a set of local affine transformations to decompose the overall articulated body deformation.
//! However, it cannot handle occlusion or out-of-plane rotation
//! well. The pose attention transfer network (PATN) (Zhu et al.
//! 2019) consists of an image stream and a pose stream, and it
//!
//!
//! --- PAGE BREAK ---
//! uses an attention mask to enhance the feature maps. However, it only processes features locally, and there is no explicit geometric manipulation or appearance transfer of the
//! source image. By contrast, our 2s-ATN consists of a source
//! stream and a target stream and explicitly finds their spatial
//! correspondence to perform appearance transfer.
//! Most recently, (Tang et al. 2019) proposes a cycle-incycle GAN, which is a cross-modal framework exploring
//! joint exploitation of the keypoint and the image data in an
//! interactive manner. (Ren et al. 2020) introduces a differentiable global-flow local-attention framework to reassemble the inputs at the feature level. (Men et al. 2020) proposes the attribute-decomposed GAN, which means to embed human attributes into the latent space as independent
//! codes and thus achieve flexible and continuous control of attributes via mixing and interpolation operations in explicit
//! style representations. (Huang et al. 2020) introduces an
//! appearance-aware pose stylizer, which generates human images by coupling the target pose with the conditioned person
//! appearance progressively. (Lathuilière et al. 2020) employs
//! the local attention mechanism to select relevant information
//! from multi-source human images for human image generation. RATE-Net (Yang et al. 2020) leverages an additional
//! texture enhancing module to extract appearance information
//! from the source image and estimate a fine-grained residual
//! texture map. This helps refine the coarse estimation from the
//! pose transfer module. (Gao et al. 2020) proposes a portrait
//! photo recapture system with two modules that complement
//! each other from both intra-part and inter-part perspectives to
//! easily transform their portraits to the desired posture.
//! Several approaches adopt DensePose (Alp Güler,
//! Neverova, and Kokkinos 2018), 3D pose (Li, Huang, and
//! Loy 2019), or human parsing (Dong et al. 2018) to generate
//! person images since they contain more information, e.g.,
//! the body part segmentation or depth. However, the keypoint
//! based pose representation is much cheaper to obtain and
//! more flexible. Therefore, we prefer to use a keypoint-based
//! representation.
//!
//! 2.2
//!
//! Self-attention
//!
//! The self-attention (Cheng, Dong, and Lapata 2016; Parikh
//! et al. 2016) was first introduced for natural language processing. It calculates the response of a certain position in the
//! sequence by paying attention to all positions in the same sequence. (Vaswani et al. 2017) proves that the machine translation model could obtain the state-of-the-art results using
//! the self-attention. (Parmar et al. 2018) introduces an image
//! transformer model that adds the self-attention to an automatic regression model for image generation. (Wang et al.
//! 2018) formulates the self-attention as a non-local operation to model the spatial-temporal dependencies in video sequences. (Zhang et al. 2019) proposes a self-attention GAN
//! enforcing the generator to gradually consider non-local relationships in the feature space. It can learn to find long-range
//! dependencies within internal representations of images.
//! Although our appearance transfer module (AT-module) is
//! inspired by the self-attention, they are significantly different.
//! The queries, keys and values in the AT-module are specially
//! designed for pose-guided appearance transfer, and they are
//!
//! semantically different. By contrast, these items in the selfattention are obtained from the same input. As a result, the
//! self-attention models the non-local relations within a single
//! feature map while our AT-module finds the spatial correspondence between the source stream and the target stream
//! to perform appearance transfer.
//!
//! 3
//! 3.1
//!
//! Our Approach
//!
//! Overview of 2s-ATN
//!
//! As illustrated in Fig. 2, our 2s-ATN is a multi-stage architecture consisting of two streams. The input of the target stream
//! is the target pose Pt . The input of the source stream is the
//! concatenation of the source pose Ps and the source image
//! Is . Both source and target poses are represented as keypoint
//! heatmaps. The output of the network is a generated target
//! image It containing the same person as the source image Is
//! but in the target pose Pt .
//! The network first uses two encoders to produce initial feature maps for the two streams. Each encoder consists of
//! two down-sampling convolutional layers, and they do not
//! share weights. The initial source features contain both appearance and structure information while the initial target
//! features contain only structure information. Then, a cascade
//! of AT-blocks progressively transfer the appearance from the
//! source stream to the target stream guided by the structure information. Each block consists of an AT-module and several
//! feature fusion modules. That is, they have the same architecture but do not share weights. Finally, the target feature map
//! from the last AT-block passes through a decoder to generate
//! the target image. The decoder consists of two deconvolutional layers. We will detail the AT-block in Sec. 3.2 and the
//! loss function in Sec. 3.3.
//!
//! 3.2
//!
//! Appearance Transfer Block (AT-Block)
//!
//! As shown in Fig. 2, an AT-block takes as input the twostream feature maps Fs ∈ RC×H×W and Ft ∈ RC×H×W
//! obtained from the previous block or the encoder and outputs their updated feature maps F0s ∈ RC×H×W and F0t ∈
//! RC×H×W . Here C, H and W respectively denote the channels, height and width of a feature map, and the subscripts s
//! and t respectively indicate the source and target streams. An
//! AT-block consists of an AT-module and several two-stream
//! feature fusion modules, which are detailed below.
//! Appearance transfer module (AT-module). The
//! pipeline of an AT-module is illustrated in Fig. 3. We first
//! pass the two-stream feature maps Fs and Ft through
//! convolutions and reshape the results as S ∈ RC×HW and
//! T ∈ RC×HW , respectively. Then we feed them into 1 × 1
//! convolution layers (implemented as matrix multiplications)
//! to produce three matrices K ∈ RC̄×HW , V ∈ RĈ×HW and
//! Q ∈ RC̄×HW :
//! K = Wk S
//! V = Wv S
//! Q = Wq T
//!
//! (1)
//! (2)
//! (3)
//!
//! where Wk ,Wq ∈ RC̄×C , Wv ∈ RĈ×C are learnable
//! weight matrices. We set C̄ = C/8, Ĉ = C/2 for mem-
//!
//!
//! --- PAGE BREAK ---
//! Figure 3: Illustration of the proposed appearance transfer module (AT-module). It calculates the correspondence between the
//! feature vector at each location in the target stream and the feature vector at each location in the source stream. Then the
//! correspondence is used to transfer the appearance information from the source stream to the target stream.
//! ory efficiency, and it does not cause a significant performance drop. Each column of K, V or Q is a key, a value or
//! a query respectively. Our AT-module means to match (target) queries to the (source) keys and then use the correspondence to transfer the relevant (source) values from the source
//! stream to the target stream.
//! To achieve this goal, we first obtain a correspondence map
//! D ∈ RHW ×HW by applying a softmax normalization to
//! each row of QT K:
//! exp(QTi Kj )
//! Dij = PHW
//! T
//! j=1 exp(Qi Kj )
//!
//! (4)
//!
//! where Dij is the (i, j)th element of D, Qi is the ith column
//! of Q, Kj is the jth column of K. Dij is a soft correspondence score between the ith query, i.e., the ith position in the
//! source feature map, and the jth key, i.e., the jth position in
//! the target feature map. We can interpret the ith row of D as a
//! probability distribution of each key matching the ith query.
//! The correspondence map serves as the basis of appearance
//! transfer.
//! Then we retrieve the value for the ith query as a linear
//! combination of the columns of V weighted by the ith row
//! of D. A matrix Wo ∈ RC×Ĉ is multiplied to the retrieved
//! values to increase their dimension:
//! A = Wo VDT
//!
//! (5)
//!
//! where A ∈ RC×HW is the appearance information to be
//! transferred from the source stream to the target stream. During the query-and-transfer process, the source appearance
//! is aligned with the target pose. Since the alignment is nonlocal, our AT-module can handle large pose transform.
//! Not all content of the target image can be found in the
//! source image because of occlusion. To encourage the target
//! stream to generate new content that can not be found in the
//!
//! source stream, we update A by multiplying it with a scale
//! parameter and adding back the target stream T:
//! A0 = σA + T
//!
//! (6)
//!
//! where σ is a learnable scalar.
//! Two-stream feature fusion modules. As shown in Fig.
//! 2, the features in the target stream are updated by fusing
//! the features in the source stream and the transferred appearance. We also add a residual connection (He et al. 2016) to
//! ease the training process. The features in the source stream
//! are updated by fusing the new target features. Two common
//! choices of the fusion function are summation and concatenation. Our ablation study indicates the latter works better for
//! both fusion modules. The two-stream feature fusion modules are important as they allow local information exchange
//! between the two streams, which supplements the non-local
//! appearance transfer.
//!
//! 3.3
//!
//! Loss Function
//!
//! The full loss function is:
//! L = arg min max αg LGAN + α1 L1 + αp Lp
//! G
//!
//! D
//!
//! (7)
//!
//! where LGAN , L1 and Lp respectively denote the adversarial loss, the `1 -norm loss and the perceptual loss, and αg ,
//! α1 and αp represent their respective weights. L1 calculates
//! the `1 -norm distance between the generated image It and the
//! ground truth target image Igt : `1 = ||Igt −It ||1 . The perceptual loss Lp has been widely used for image generation and
//! translation (Esser, Sutter, and Ommer 2018; Siarohin et al.
//! 2018; Ledig et al. 2017; Johnson, Alahi, and Fei-Fei 2016)
//! as it helps generate more realistic and smoother images. It is
//! defined as:
//! 1
//! ||φρ (Igt ) − φρ (It )||1
//! (8)
//! Lp =
//! Wρ Hρ Cρ
//!
//!
//! --- PAGE BREAK ---
//! Figure 4: Qualitative comparison on Market-1501 and DeepFashion.
//! where φρ is the output of the conv1 2 layer from the VGG19 model (Simonyan and Zisserman 2014) pretrained on ImageNet (Russakovsky et al. 2015), and Wρ , Hρ , Cρ are the
//! width, height and depth of φρ , respectively. We adopt the
//! adversarial loss introduced in (Zhu et al. 2019). It consists
//! of an appearance discriminator and a shape discriminator to
//! determine the possibility that the generated image contains
//! the same person in the input image and the degree to which
//! the generated image is aligned with the target pose.
//!
//! 4
//!
//! Experiment
//!
//! In this section, we conduct extensive experiments both qualitatively and quantitatively. They will demonstrate that our
//! 2s-ATN outperforms state-of-the-art methods regarding visual fidelity and alignment with targeted person poses.
//! Datasets. We use two challenging person image datasets:
//! Market-1501 (Zheng et al. 2015) and DeepFashion (Liu
//! et al. 2016). The resolution of images in DeepFashion is
//! higher (256 × 256) than images in Market-1501 (128 × 64).
//! We employ OpenPose (Cao et al. 2017) to detect human
//! body joints. Both the source and target poses consist of an
//! 18-channel heatmap encoding the positions of 18 human
//! body joints . We have 263,632 pairs of training images from
//! Market-1501, and 101,966 pairs from DeepFashion. Their
//! testing sets contain 12,000 pairs and 8,570 pairs, respectively. Note the person identities of the training set do not
//! overlap with those of the testing set.
//! Evaluation metrics. We follow (Ma et al. 2017; Siarohin
//! et al. 2018; Zhu et al. 2019) and adopt Structure Similarity
//! (SSIM) (Wang et al. 2004), Inception Score (IS) (Salimans
//! et al. 2016), and their masked versions, i.e., Mask-SSIM and
//! Mask-IS, as the evaluation metrics. Moreover, we adopt the
//!
//! PCKh score proposed in (Zhu et al. 2019) to assess the shape
//! consistency explicitly.
//! Implement details. Our method is implemented in PyTorch using two NVIDIA GeForce RTX 2080 Ti GPUs.
//! The Adam optimizer (Kingma and Ba 2014) is adopted to
//! train the proposed model for around 90k iterations with
//! β1 = 0.5, β2 = 0.999. The learning rate is fixed as
//! 0.0001 in the first 60k iterations and then linearly decayed
//! to 0 in the last 30k iterations. We use 9 AT-blocks in
//! the generator for both datasets. For the hyper-parameters,
//! (αg ,α1 ,αp ) are set as (20, 17, 17) for DeepFashion and (20,
//! 17, 17) for Market-1501, respectively. Instance normalization (Ulyanov, Vedaldi, and Lempitsky 2016) is applied for
//! both datasets. The batch size is set as 7 for DeepFashion and
//! 32 for Market-1501. Dropout (Hinton et al. 2012) is only
//! used in the AT-blocks, and the dropout rate is set to 0.5.
//! Leaky ReLU (Maas, Hannun, and Ng 2013) is applied after every convolution or normalization layer in the discriminators, and its negative slope coefficient is set to 0.2. Each
//! training epoch takes about 95 seconds for Market-1501 and
//! 460 seconds for Deepfashion. In total, it takes about one day
//! to train the network on Market-1501 and four days on DeepFashion.
//!
//! 4.1
//!
//! Comparison with States-of-the-art Methods
//!
//! Quantitative and qualitative comparison. We compare the
//! proposed 2s-ATN with several state-of-the-art methods such
//! as DPIG (Ma et al. 2018), VUnet (Esser, Sutter, and Ommer 2018), Deform (Siarohin et al. 2018), PATN (Zhu et al.
//! 2019), BTF (AlBahar and Huang 2019), C2GAN (Tang et al.
//! 2019), ADG (Men et al. 2020) and APS (Huang et al. 2020).
//! Table 1 shows the quantitative results measured by SSIM,
//!
//!
//! --- PAGE BREAK ---
//! Method
//! DPIG (Ma et al. 2018)
//! VUNet (Esser, Sutter, and Ommer 2018)
//! Deform (Siarohin et al. 2018)
//! PATN (Zhu et al. 2019)
//! BFT (AlBahar and Huang 2019)
//! C2GAN (Tang et al. 2019)
//! ADG (Men et al. 2020)
//! APS (Huang et al. 2020)
//! PATN* (Zhu et al. 2019)
//! Ours
//! Real Data
//!
//! SSIM
//! 0.099
//! 0.266
//! 0.290
//! 0.311
//! −
//! 0.282
//! −
//! 0.312
//! 0.301
//! 0.320
//! 1.000
//!
//! IS
//! 3.483
//! 2.965
//! 3.185
//! 3.323
//! −
//! 3.349
//! −
//! 3.132
//! 3.344
//! 3.504
//! 3.890
//!
//! Market-1501
//! Mask-SSIM Mask-IS
//! 0.614
//! 3.491
//! 0.793
//! 3.549
//! 0.805
//! 3.502
//! 0.811
//! 3.773
//! −
//! −
//! 0.811
//! 3.510
//! −
//! −
//! 0.808
//! 3.729
//! 0.805
//! 3.773
//! 0.813
//! 3.845
//! 1.000
//! 3.706
//!
//! PCKh
//! −
//! 0.92
//! −
//! 0.94
//! −
//! −
//! −
//! 0.94
//! 0.94
//! 0.94
//! 1.00
//!
//! DeepFashion
//! SSIM
//! IS
//! PCKh
//! 0.614
//! 3.228
//! −
//! 0.763 3.440
//! 0.93
//! 0.756
//! 3.439
//! −
//! 0.773
//! 3.209
//! 0.96
//! 0.767
//! 3.220
//! −
//! −
//! −
//! −
//! 0.772
//! 3.364
//! −
//! 0.775
//! 3.295
//! 0.96
//! 0.767
//! 3.209
//! 0.96
//! 0.775 3.206
//! 0.96
//! 1.000
//! 4.053
//! 1.00
//!
//! Table 1: Quantitative results on Market-1501 and DeepFashion. (∗) denotes the results reproduced by us. For all metrics, higher
//! values indicate better performance.
//! Method
//! PG2 (Ma et al. 2017)
//! Deform (Siarohin et al. 2018)
//! VUNet (Esser, Sutter, and Ommer 2018)
//! PATN (Zhu et al. 2019)
//! Ours
//!
//! #Parameters
//! 437.09 M
//! 82.08 M
//! 139.36 M
//! 41.36 M
//! 43.28 M
//!
//! Speed
//! 10.36 fps
//! 17.74 fps
//! 29.37 fps
//! 60.61 fps
//! 74.21fps
//!
//! Table 2: Comparison of model sizes and inference speeds on DeepFashion.
//! IS, Mask-SSIM, Mask-IS, and PCKh metrics. The 2s-ATN
//! achieves the best performance under most metrics on the two
//! datasets. Fig. 4 shows that the appearance or texture generated by the proposed method is more consistent and appealing than the others.
//! We also take PATN (Zhu et al. 2019) as a strong baseline and reproduce its results via the code provided by the
//! authors. Compared with PATN, 2s-ATN has achieved better
//! performance under the SSIM, IS, Mask-SSIM, and MaskIS metrics and the same performance under PCKh. However, we observe that compared with the images generated
//! by PATN (Zhu et al. 2019), the images generated by 2s-ATN
//! are more realistic and have fewer visual artifacts (as Fig. 4
//! depicted).
//! Comparison of model size and inference speed. Tab. 2
//! compares the model size and (single-GPU) inference speed
//! of our 2s-ATN with those of four state-of-the-art methods.
//! The GPU time is reported. Thanks to the simple and clean
//! structure of our 2s-ATN, it achieves the highest processing
//! speed with a much smaller or comparable model size.
//!
//! 4.2
//!
//! Ablation Study
//!
//! In this section, we perform ablation study to analyze the impact of each component in our model on the performance.
//! There are two kinds of essential modules in the 2s-ATN: the
//! AT-module and the two-stream feature fusion modules.
//! Effect of the AT-module. The results of this ablation
//! study are shown in Tab. 3 and Fig. 5. Removing the ATmodules from our 2s-ATN will decrease SSIM, IS, MaskSSIM and Mask-IS from 0.320, 3.504, 0.813, and 3.845 to
//! 0.313, 3.252, 0.812, and 3.793, respectively. The qualitative
//!
//! Method
//! 2s-ATN (Full)
//! w/o AT-module
//!
//! SSIM
//! 0.320
//! 0.313
//!
//! Market-1501
//! IS
//! Mask-SSIM
//! 3.504
//! 0.813
//! 3.252
//! 0.812
//!
//! Mask-IS
//! 3.845
//! 3.793
//!
//! Table 3: Ablation study on the AT-module.
//! Method
//! T:cat, S:cat
//! T:add, S:cat
//! T:cat, S:add
//! T:add, S:add
//!
//! SSIM
//! 0.320
//! 0.320
//! 0.309
//! 0.234
//!
//! Market-1501
//! IS
//! Mask-SSIM
//! 3.504
//! 0.813
//! 3.335
//! 0.813
//! 3.460
//! 0.809
//! 3.894
//! 0.768
//!
//! Mask-IS
//! 3.845
//! 3.815
//! 3.843
//! 3.675
//!
//! Table 4: Ablation study on the two-stream feature fusion
//! modules. “T” and “S” denote the target stream and source
//! stream, respectively.
//!
//! results show the images generated by the full model look
//! much better than those generated by the model without the
//! AT-module. Thus the proposed AT-module helps generate
//! photo-realistic person images as it enables the network to
//! perform non-local spatial manipulation.
//! Effect of two-stream feature fusion modules. Qualitative and quantitative results are shown in Fig. 5 and Tab. 4.
//! “T” and “S” denote the two fusion modules in the target
//! stream and the source stream, respectively. “Cat” and “add”
//! are short for concatenation and summation, respectively.
//! The quantitative results in Tab. 4 indicate concatenation generally works better than summation for both fusion modules.
//!
//!
//! --- PAGE BREAK ---
//! Figure 5: Ablation study of the proposed 2s-ATN on Market-1501. First column: input and ground truth output. Second column
//! (orange): ablation study on the number of AT-blocks. Third column (red): ablation study on the AT-module. Fourth column
//! (green): ablation study on the two-stream feature fusion modules.
//!
//! #AT-blocks
//! 5
//! 7
//! 9
//! 11
//! 13
//!
//! SSIM
//! 0.317
//! 0.319
//! 0.320
//! 0.314
//! 0.316
//!
//! Market-1501
//! IS
//! Mask-SSIM
//! 3.354
//! 0.813
//! 3.490
//! 0.812
//! 3.504
//! 0.813
//! 3.462
//! 0.810
//! 3.323
//! 0.813
//!
//! Mask-IS
//! 3.783
//! 3.799
//! 3.845
//! 3.844
//! 3.852
//!
//! Table 5: Ablation study on the number of AT-blocks.
//!
//! The visualization in Fig. 5 shows using concatenation in
//! both streams can help our 2s-ATN generate more appealing images than other combinations of fusion methods. The
//! shape and appearance of generated persons are more consistent with those in the ground truth.
//! Effect of the number of AT-blocks. Quantitative and
//! qualitative results are shown in Tab. 5 and Fig. 5. We observe that the proposed generator works best when it consists of 9 AT-blocks. Increasing or decreasing the number of
//! AT-blocks may result in slightly poor quantitative and qualitative performance. Therefore, we have used 9 AT-blocks as
//!
//! the default setting in the other experiments. It is worth noting that using only 5 AT-blocks still results in acceptable,
//! through not the best, results both qualitatively and quantitatively. This indicates that a light version of the proposed
//! method is still applicable when the computation resource is
//! limited.
//!
//! 5
//!
//! Conclusion
//!
//! This paper introduces a two-stream appearance transfer network (2s-ATN) for pose guided person image generation.
//! It is a novel multi-stage architecture consisting of a source
//! stream and a target stream. Its core is a novel appearance
//! transfer module (AT-module) in each stage. It learns to find
//! the structure correspondence between the two-stream feature maps and then transfer the appearance information from
//! the source stream to the target stream. We also supplement
//! the non-local appearance transfer with the local two-stream
//! feature fusion modules. Experimental results indicate that
//! the proposed 2s-ATN can effectively handle large pose deformation and occlusion while retaining the texture details.
//!
//!
//! --- PAGE BREAK ---
//! References
//! AlBahar, B.; and Huang, J.-B. 2019. Guided image-to-image
//! translation with bi-directional feature transformation. In
//! Proceedings of the IEEE International Conference on Computer Vision, 9016–9025.
//! Alp Güler, R.; Neverova, N.; and Kokkinos, I. 2018. Densepose: Dense human pose estimation in the wild. In Proceedings of the IEEE Conference on Computer Vision and
//! Pattern Recognition, 7297–7306.
//! Cao, Z.; Simon, T.; Wei, S.-E.; and Sheikh, Y. 2017. Realtime multi-person 2d pose estimation using part affinity
//! fields. In Proceedings of the IEEE conference on computer
//! vision and pattern recognition, 7291–7299.
//! Cheng, J.; Dong, L.; and Lapata, M. 2016. Long shortterm memory-networks for machine reading. arXiv preprint
//! arXiv:1601.06733 .
//! Dong, H.; Liang, X.; Gong, K.; Lai, H.; Zhu, J.; and Yin,
//! J. 2018. Soft-gated warping-gan for pose-guided person image synthesis. In Advances in neural information processing
//! systems, 474–484.
//! Esser, P.; Sutter, E.; and Ommer, B. 2018. A variational
//! u-net for conditional appearance and shape generation. In
//! Proceedings of the IEEE Conference on Computer Vision
//! and Pattern Recognition, 8857–8866.
//! Gao, C.; Liu, S.; He, R.; Yan, S.; and Li, B. 2020. Recapture
//! as You Want. arXiv preprint arXiv:2006.01435 .
//! Goodfellow, I.; Pouget-Abadie, J.; Mirza, M.; Xu, B.;
//! Warde-Farley, D.; Ozair, S.; Courville, A.; and Bengio, Y.
//! 2014. Generative adversarial nets. In Advances in neural
//! information processing systems, 2672–2680.
//! Han, X.; Hu, X.; Huang, W.; and Scott, M. R. 2019. Clothflow: A flow-based model for clothed person generation. In
//! Proceedings of the IEEE International Conference on Computer Vision, 10471–10480.
//! He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep residual learning for image recognition. In Proceedings of the
//! IEEE conference on computer vision and pattern recognition, 770–778.
//! Hinton, G. E.; Srivastava, N.; Krizhevsky, A.; Sutskever, I.;
//! and Salakhutdinov, R. R. 2012. Improving neural networks
//! by preventing co-adaptation of feature detectors. arXiv
//! preprint arXiv:1207.0580 .
//! Huang, S.; Xiong, H.; Cheng, Z.-Q.; Wang, Q.; Zhou, X.;
//! Wen, B.; Huan, J.; and Dou, D. 2020. Generating Person Images with Appearance-aware Pose Stylizer. arXiv preprint
//! arXiv:2007.09077 .
//! Isola, P.; Zhu, J.-Y.; Zhou, T.; and Efros, A. A. 2017. Imageto-image translation with conditional adversarial networks.
//! In Proceedings of the IEEE conference on computer vision
//! and pattern recognition, 1125–1134.
//! Johnson, J.; Alahi, A.; and Fei-Fei, L. 2016. Perceptual losses for real-time style transfer and super-resolution.
//! In European conference on computer vision, 694–711.
//! Springer.
//!
//! Kingma, D. P.; and Ba, J. 2014. Adam: A method for
//! stochastic optimization. arXiv preprint arXiv:1412.6980 .
//! Kingma, D. P.; and Welling, M. 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 .
//! Lathuilière, S.; Sangineto, E.; Siarohin, A.; and Sebe, N.
//! 2020. Attention-based Fusion for Multi-source Human Image Generation. In The IEEE Winter Conference on Applications of Computer Vision, 439–448.
//! LeCun, Y.; Bottou, L.; Bengio, Y.; and Haffner, P. 1998.
//! Gradient-based learning applied to document recognition.
//! Proceedings of the IEEE 86(11): 2278–2324.
//! Ledig, C.; Theis, L.; Huszár, F.; Caballero, J.; Cunningham,
//! A.; Acosta, A.; Aitken, A.; Tejani, A.; Totz, J.; Wang, Z.;
//! et al. 2017. Photo-realistic single image super-resolution using a generative adversarial network. In Proceedings of the
//! IEEE conference on computer vision and pattern recognition, 4681–4690.
//! Li, Y.; Huang, C.; and Loy, C. C. 2019. Dense intrinsic appearance flow for human pose transfer. In Proceedings of the
//! IEEE Conference on Computer Vision and Pattern Recognition, 3693–3702.
//! Liu, W.; Piao, Z.; Min, J.; Luo, W.; Ma, L.; and Gao, S.
//! 2019. Liquid warping GAN: A unified framework for human motion imitation, appearance transfer and novel view
//! synthesis. In Proceedings of the IEEE International Conference on Computer Vision, 5904–5913.
//! Liu, Z.; Luo, P.; Qiu, S.; Wang, X.; and Tang, X. 2016. Deepfashion: Powering robust clothes recognition and retrieval
//! with rich annotations. In Proceedings of the IEEE conference on computer vision and pattern recognition, 1096–
//! 1104.
//! Ma, L.; Jia, X.; Sun, Q.; Schiele, B.; Tuytelaars, T.; and
//! Van Gool, L. 2017. Pose guided person image generation.
//! In Advances in neural information processing systems, 406–
//! 416.
//! Ma, L.; Sun, Q.; Georgoulis, S.; Van Gool, L.; Schiele, B.;
//! and Fritz, M. 2018. Disentangled person image generation.
//! In Proceedings of the IEEE Conference on Computer Vision
//! and Pattern Recognition, 99–108.
//! Maas, A. L.; Hannun, A. Y.; and Ng, A. Y. 2013. Rectifier
//! nonlinearities improve neural network acoustic models. In
//! Proc. icml, volume 30, 3.
//! Men, Y.; Mao, Y.; Jiang, Y.; Ma, W.-Y.; and Lian, Z.
//! 2020. Controllable person image synthesis with attributedecomposed gan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5084–
//! 5093.
//! Mirza, M.; and Osindero, S. 2014. Conditional generative
//! adversarial nets. arXiv preprint arXiv:1411.1784 .
//! Parikh, A. P.; Täckström, O.; Das, D.; and Uszkoreit, J.
//! 2016. A decomposable attention model for natural language
//! inference. arXiv preprint arXiv:1606.01933 .
//! Parmar, N.; Vaswani, A.; Uszkoreit, J.; Kaiser, Ł.; Shazeer,
//! N.; Ku, A.; and Tran, D. 2018. Image transformer. arXiv
//! preprint arXiv:1802.05751 .
//!
//!
//! --- PAGE BREAK ---
//! Qian, X.; Fu, Y.; Xiang, T.; Wang, W.; Qiu, J.; Wu, Y.; Jiang,
//! Y.-G.; and Xue, X. 2018. Pose-normalized image generation
//! for person re-identification. In Proceedings of the European
//! conference on computer vision (ECCV), 650–667.
//! Ren, Y.; Yu, X.; Chen, J.; Li, T. H.; and Li, G. 2020. Deep
//! image spatial transformation for person image generation.
//! In Proceedings of the IEEE/CVF Conference on Computer
//! Vision and Pattern Recognition, 7690–7699.
//! Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U-net: Convolutional networks for biomedical image segmentation. In
//! International Conference on Medical image computing and
//! computer-assisted intervention, 234–241. Springer.
//! Russakovsky, O.; Deng, J.; Su, H.; Krause, J.; Satheesh, S.;
//! Ma, S.; Huang, Z.; Karpathy, A.; Khosla, A.; Bernstein, M.;
//! et al. 2015. Imagenet large scale visual recognition challenge. International journal of computer vision 115(3): 211–
//! 252.
//! Salimans, T.; Goodfellow, I.; Zaremba, W.; Cheung, V.; Radford, A.; and Chen, X. 2016. Improved techniques for training gans. In Advances in neural information processing systems, 2234–2242.
//! Siarohin, A.; Sangineto, E.; Lathuiliere, S.; and Sebe, N.
//! 2018. Deformable gans for pose-based human image generation. In Proceedings of the IEEE Conference on Computer
//! Vision and Pattern Recognition, 3408–3416.
//! Simonyan, K.; and Zisserman, A. 2014. Very deep convolutional networks for large-scale image recognition. arXiv
//! preprint arXiv:1409.1556 .
//! Tang, H.; Xu, D.; Liu, G.; Wang, W.; Sebe, N.; and Yan,
//! Y. 2019. Cycle in cycle generative adversarial networks for
//! keypoint-guided image generation. In Proceedings of the
//! 27th ACM International Conference on Multimedia, 2052–
//! 2060.
//! Ulyanov, D.; Vedaldi, A.; and Lempitsky, V. 2016. Instance
//! normalization: The missing ingredient for fast stylization.
//! arXiv preprint arXiv:1607.08022 .
//! Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones,
//! L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. In Advances in neural information
//! processing systems, 5998–6008.
//! Wang, X.; Girshick, R.; Gupta, A.; and He, K. 2018. Nonlocal neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, 7794–
//! 7803.
//! Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P.
//! 2004. Image quality assessment: from error visibility to
//! structural similarity. IEEE transactions on image processing 13(4): 600–612.
//! Yang, C.; Wang, Z.; Zhu, X.; Huang, C.; Shi, J.; and Lin, D.
//! 2018. Pose guided human video generation. In Proceedings
//! of the European Conference on Computer Vision (ECCV),
//! 201–216.
//! Yang, L.; Wang, P.; Zhang, X.; Wang, S.; Gao, Z.; Ren,
//! P.; Xie, X.; Ma, S.; and Gao, W. 2020. Region-adaptive
//!
//! texture enhancement for detailed person image synthesis.
//! In 2020 IEEE International Conference on Multimedia and
//! Expo (ICME), 1–6. IEEE.
//! Zhang, H.; Goodfellow, I.; Metaxas, D.; and Odena, A.
//! 2019. Self-attention generative adversarial networks. In International Conference on Machine Learning, 7354–7363.
//! Zheng, L.; Shen, L.; Tian, L.; Wang, S.; Wang, J.; and Tian,
//! Q. 2015. Scalable person re-identification: A benchmark. In
//! Proceedings of the IEEE international conference on computer vision, 1116–1124.
//! Zhu, J.-Y.; Park, T.; Isola, P.; and Efros, A. A. 2017. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international
//! conference on computer vision, 2223–2232.
//! Zhu, Z.; Huang, T.; Shi, B.; Yu, M.; Wang, B.; and Bai, X.
//! 2019. Progressive pose attention transfer for person image
//! generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2347–2356.
//!
//!
//! --- PAGE BREAK ---
//!
//! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//!
