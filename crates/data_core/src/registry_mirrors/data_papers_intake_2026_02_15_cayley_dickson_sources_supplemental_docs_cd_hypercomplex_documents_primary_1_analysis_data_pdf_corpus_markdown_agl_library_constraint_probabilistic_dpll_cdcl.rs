//! # Extracted text: dpll_cdcl.pdf
//!
//! - source_root: `/home/eirikr/Documents/AGL_Library/Constraint_Probabilistic_Documentation`
//! - source_relpath: `dpll_cdcl.pdf`
//! - source_abs: `/home/eirikr/Documents/AGL_Library/Constraint_Probabilistic_Documentation/dpll_cdcl.pdf`
//! - detected_kind: `pdf`
//! - extracted_at_utc: `2026-01-02T17:30:43+00:00`
//! - pages: `3`
//! - title: `Neural Code Search Evaluation Dataset`
//! - author: `Hongyu Li, Seohyun Kim, and Satish Chandra`
//! - subject: ``
//! - keywords: ``
//! - creation_date: `Wed Oct  2 17:33:55 2019 PDT`
//! - mod_date: `Wed Oct  2 17:33:55 2019 PDT`
//! - encrypted: `no`
//!
//! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~text
//! arXiv:1908.09804v6 [cs.SE] 2 Oct 2019
//!
//! Neural Code Search Evaluation Dataset
//! Hongyu Li
//!
//! Seohyun Kim
//!
//! Satish Chandra
//!
//! Facebook, Inc.
//! U.S.A
//! hongyul@fb.com
//!
//! Facebook, Inc.
//! U.S.A
//! skim131@fb.com
//!
//! Facebook, Inc.
//! U.S.A
//! satch@fb.com
//!
//! Abstract
//!
//! 2 Dataset Contents
//!
//! There has been an increase of interest in code search using natural language. Assessing the performance of such
//! code search models can be diﬃcult without a readily available evaluation suite. In this paper, we present an evaluation
//! dataset consisting of natural language query and code snippet pairs, with the hope that future work in this area can use
//! this dataset as a common benchmark. We also provide the
//! results of two code search models ([6] and [1]) from recent
//! work as a benchmark.
//!
//! In this section, we explain what data we are releasing.
//!
//! 1
//!
//! Introduction
//!
//! In recent years, learning the mapping between natural language and code snippets has been a popular ﬁeld of research.
//! In particular, [6], [1], [2] have explored ﬁnding relevant code
//! snippets given a natural language query, with the models
//! varying from using word embeddings and IR techniques to
//! using sophisticated neural networks. To evaluate the performance of these models, Stack Overﬂow questions and
//! code answer pairs are prime candidates, as Stack Overﬂow
//! questions well resemble what a developer may ask. Such an
//! example is "Close/hide the Android Soft Keyboard".12 One
//! of the ﬁrst answers34 on Stack Overﬂow correctly answers
//! this question. However, collecting these questions can be
//! tedious, and systematically comparing various models can
//! pose a challenge.
//! To this end, we have constructed an evaluation dataset,
//! which contains natural language queries and relevant code
//! snippet answers from Stack Overﬂow. It also includes code
//! snippet examples from the search corpus (public repositories from GitHub) that correctly answers each query. We
//! hope that this dataset can be served as a benchmark to evaluate performance across various code search models.
//! The paper is organized as follows. First we will explain
//! what data we are releasing in the dataset. Then we will describe the process for obtaining this dataset. Finally, we will
//! evaluate two code search models of our own creation, NCS
//! and UNIF, on the evaluation dataset as a benchmark.
//! 1 https://stackoverflow.com/questions/1109022/
//!
//! close-hide-the-android-soft-keyboard
//! 2 Author: Vidar Vestnes.
//! https://stackoverflow.com/users/133858/vidar-vestnes
//! 3 https://stackoverflow.com/a/1109108
//! 4 Author: Reto Meier. https://stackoverflow.com/users/822/reto-meier
//!
//! 2.1 GitHub Repositories
//! The most popular Android repositories on GitHub (ranked
//! by the number of stars) is used to create the search corpus.
//! For each repository that we indexed, we provide the link,
//! speciﬁc to the commit that was used.5 In total, there are
//! 24,549 repositories.6 We will release a text ﬁle containing
//! the download links for these GitHub repositories. See Listing 1 for an example.
//! 2.2 Search Corpus
//! The search corpus is indexed using all method bodies parsed
//! from the 24,549 GitHub repositories. In total, there are 4,716,814
//! methods in this corpus. The code search model will ﬁnd relevant code snippets (i.e. method bodies) from this corpus
//! given a natural language query. In this data release, we will
//! provide the following information for each method in the
//! corpus:
//! • id: Each method in the corpus has a unique numeric identiﬁer. This ID number will also be referenced in our evaluation
//! dataset.
//! • ﬁlepath: The ﬁle path is in the format of
//! :owner/:repo/relative-file-path-to-the-repo
//!
//! • method_name
//! • start_line: Starting line number of the method in the ﬁle.
//! • end_line: Ending line number of the method in the ﬁle.
//! • url: GitHub link to the method body with commit ID and line
//! numbers encoded.
//!
//! Listing 2 provides an example of a method in the search corpus.
//! 2.3 Evaluation Dataset
//! The evaluation dataset is composed of 287 Stack Overﬂow
//! question and answer pairs, for which we release the following information:
//! • stackoverﬂow_id: Stack Overﬂow post ID.
//! • question: Title of the Stack Overﬂow post.
//! • question_url: URL of the Stack Overﬂow post.
//! • answer: Code snippet answer to the question.
//! 5 From August 2018
//! 6 There were originally 26,109 repositories - the diﬀerence is due to reasons
//!
//! outside of our control (e.g. repositories getting deleted). Note that not all
//! of the links in this dataset may not always be available in the future for the
//! similar reasons.
//!
//!
//! --- PAGE BREAK ---
//! Conference’17, July 2017, Washington, DC, USA
//!
//! Li, Kim, and Chandra
//!
//! https://github.com/00-00-00/ably-chat/archive/9bb2e36acc24f1cd684ef5d1b98d837055ba9cc8.zip
//! https://github.com/01sadra/Detoxiom/archive/c3fffd36989b0cd93bd09cbaa35123b9d605f989.zip
//! https://github.com/0411ameya/MPG_update/archive/27ac5531ca2c2f123e0cb854ebcb4d0441e2bc98.zip
//! https://github.com/0508994/MinesweeperGO/archive/ba0e0e45d2da21dde2365ce09277aad511de6885.zip
//! https://github.com/07101994/My-PPT-Presentation/archive/b89b17a962d5c3e5682fa751228a9f9ca593d77b.zip
//! https://github.com/0912718/ICT-lab/archive/d1d723edb722013cc83761f0f9df252cfd3361c3.zip
//! https://github.com/0Cubed/ZeroMediaPlayer/archive/d84c675f9dc8b16f823bb252db9ee368fbd5cd8e.zip
//! ...
//!
//! Listing 1. GitHub repositories download links example.
//!
//! 2.4 NCS / UNIF Score Sheet
//!
//! {
//! "id": 4716813,
//! "filepath": "Mindgames/VideoStreamServer/playersdk/src
//! /main/java/com/kaltura/playersdk/
//! PlayerViewController.java",
//! "method_name": "notifyKPlayerEvent",
//! "start_line": 506,
//! "end_line": 566,
//! "url": "https://github.com/Mindgames/VideoStreamServer
//! /blob/b7c73d2bcd296b3a24f83cf67d6a5998c7a1af6b/
//! playersdk/src/main/java/com/kaltura/playersdk/
//! PlayerViewController.java\#L506-L566"
//! }
//!
//! Listing 2. Search corpus example.
//!
//! {
//! "stackoverflow_id": 1109022,
//! "question": "Close/hide the Android Soft Keyboard",
//! "question_url": "https://stackoverflow.com/questions
//! /1109022/close-hide-the-android-soft-keyboard",
//! "question_author": "Vidar Vestnes",
//! "question_author_url":
//! "https://stackoverflow.com/users/133858",
//! "answer": "// Check if no view has focus:\nView view =
//! this.getCurrentFocus();\nif (view != null) {
//! InputMethodManager imm = (InputMethodManager)
//! getSystemService(Context.INPUT_METHOD_SERVICE);
//! imm.hideSoftInputFromWindow(view.getWindowToken()
//! , 0);}",
//! "answer_url": "https://stackoverflow.com/a/1109108",
//! "answer_author": "Reto Meier",
//! "answer_author_url":
//! "https://stackoverflow.com/users/822",
//! "examples": [1841045, 1800067, 1271795],
//! "examples_url": [
//! "https://github.com/alextselegidis/easyappointmentsandroid-client/blob/39f1e8...",
//! "https://github.com/zelloptt/zello-android-clientsdk/blob/87b45b6...",
//! "https://github.com/systers/conference-android/blob/
//! a67982abf54e0...",
//! ]
//! }
//!
//! Listing 3. Evaluation dataset example.
//!
//! • answer_url: URL of the Stack Overﬂow answer to the question.
//! • examples: 3 methods from the search corpus that best answer the question (most similar to the Stack Overﬂow answer).
//! • examples_url: GitHub links to the examples.
//!
//! Note that there may be more acceptable answers to each
//! question. See Listing 3 for a concrete example of an evaluation question in this dataset. The source of the question and
//! answer pairs is extracted from the Stack Exchange Network
//! [4].
//!
//! We provide the evaluation results for two code search models of our creation, each with two variations:
//! • NCS: an unsupervised model which uses word embedding derived directly from the search corpus[6].
//! • NCSpostrank : an extension of the base NCS model that
//! performs a post-pass ranking, as explained in [6].
//! • UNIFandroid, UNIFstackoverﬂow: a supervised extension of
//! the NCS model that uses a bag-of-words-based neural
//! network with attention. The supervision is learned using GitHub-Android-Train and StackOverﬂow-AndroidTrain datasets, respectively, as described in [1].
//! We provide the rank of the ﬁrst correct answer (FRank) for
//! each question in our evaluation dataset. The score sheet is
//! saved in a comma-delimited csv ﬁle as illustrated in Listing 4.
//! No.,StackOverflow ID,NCS FRank,NCS_postrank FRank,
//! UNIF_android FRank,UNIF_stackoverflow FRank
//! 1,1109022,NF,1,1,1
//! 2,4616095,17,1,31,19
//! 3,3004515,2,1,5,2
//! 4,1560788,1,4,5,1
//! 5,3423754,5,1,22,10
//! 6,1397361,NF,3,2,1
//!
//! Listing 4. Score sheet example. "NF" stands for correct answer not found
//! in the top 50 returned results.
//!
//! 3 How we Obtained the Dataset
//! In this section, we describe the procedure for how we obtained the data.
//! GitHub repositories. We obtained the information of
//! the GitHub repositories with the GitHub REST API [3], and
//! the source ﬁles were downloaded using publicly available
//! links.
//! Search corpus. The search corpus was obtained by dividing each ﬁle in the GitHub repositories by method-level
//! granularity.
//! Evaluation dataset. The benchmark questions were collected from a data dump publicly released by Stack Exchange
//! [4]. To select the set of Stack Overﬂow question and answer pairs, we created a heuristics-based ﬁltering pipeline
//! where we discarded open-ended, discussion-style questions.
//! We ﬁrst obtained the most popular 17,000 questions on Stack
//!
//!
//! --- PAGE BREAK ---
//! Neural Code Search Evaluation Dataset
//!
//! Conference’17, July 2017, Washington, DC, USA
//!
//! Table 1. Number of questions answered in the top 1, 5, 10 and MRR for
//! NCS, NCSpostrank , UNIFandroid and UNIFstackoverﬂow .
//! Model
//!
//! Answered@1
//!
//! Answered@5
//!
//! Answered@10
//!
//! MRR
//!
//! NCS
//! NCSpostrank
//! UNIFandroid
//! UNIFstackoverﬂow
//!
//! 33
//! 85
//! 25
//! 104
//!
//! 74
//! 151
//! 74
//! 164
//!
//! 98
//! 180
//! 110
//! 188
//!
//! 0.189
//! 0.4
//! 0.178
//! 0.465
//!
//! Overﬂow with “Android” and “Java” tags. The dataset is further ﬁltered with the following criteria: 1) there exists an upvoted code answer, 2) the ground truth code snippet has at
//! least one match in the search corpus. From this pipeline, we
//! were able to obtain 518 questions. Finally, we manually went
//! through these questions and ﬁltered out questions with vague
//! queries and/or code answers. The ﬁnal dataset contains 287
//! Stack Overﬂow question and answers pairs.
//! NCS / UNIF score sheet. To judge whether a method
//! body correctly answers the query, we compare how similar
//! it is to the Stack Overﬂow answer - we do this systematically using a code-to-code similarity tool, called Aroma [5].
//! Aroma gives a similarity score between two code snippets;
//! if this score is above a certain threshold (0.25 in our case),
//! we count it as success. This similarity score, aims to mimic
//! manually assessing the correctness of search results in an
//! automatic and reproducible fashion, while leaving out human judgment in the process. More details on how we chose
//! this threshold can be found in [1].
//!
//! 4 Evaluation
//! We provide the results for four models: NCS, NCSpostrank ,
//! UNIFandroid, and UNIFstackoverﬂow.
//! Table 1 reports the number of questions answered within
//! the top_n returned code snippet, where n = 1, 5, and 10 (Answered@1, 5, 10 in Table 1), as well as the Mean Reciprocal
//! Rank (MRR).
//!
//! References
//! [1] Jose Cambronero, Hongyu Li Seohyun Kim, Koushik Sen, and Satish
//! Chandra. When deep learning met code search. CoRR, abs/1905.03813,
//! 2019. URL: https://arxiv.org/abs/1905.03813, arXiv:1905.03813.
//! [2] Xiaodong Gu, Hongyu Zhang, and Sunghun Kim. Deep code search. In
//! Proceedings of the 40th International Conference on Software Engineering, pages 933–944. ACM, 2018.
//! [3] GitHub Inc. Github rest api v3. URL: https://developer.github.com/v3/
//! search/.
//! [4] Stack Exchange Inc. datastack exchange data dump, 2018. CC-BY-SA
//! 3.0. URL: https://archive.org/details/stackexchange.
//! [5] Sifei Luan, Di Yang, Celeste Barnaby, Koushik Sen, and Satish Chandra.
//! Aroma: Code recommendation via structural code search.
//! CoRR, abs/1812.01158, 2018. URL: http://arxiv.org/abs/1812.01158,
//! arXiv:1812.01158.
//! [6] Saksham Sachdev, Hongyu Li, Sifei Luan, Seohyun Kim, Koushik Sen,
//! and Satish Chandra. Retrieval on source code: a neural code search.
//! In Proceedings of the 2nd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, pages 31–41. ACM, 2018.
//!
//!
//! --- PAGE BREAK ---
//!
//! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//!
