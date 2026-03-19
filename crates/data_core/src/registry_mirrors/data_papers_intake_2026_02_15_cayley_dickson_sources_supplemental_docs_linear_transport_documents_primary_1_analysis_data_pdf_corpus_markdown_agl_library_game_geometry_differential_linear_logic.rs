//! # Extracted text: differential_linear_logic.pdf
//!
//! - source_root: `/home/eirikr/Documents/AGL_Library/Game_Geometry_Documentation`
//! - source_relpath: `differential_linear_logic.pdf`
//! - source_abs: `/home/eirikr/Documents/AGL_Library/Game_Geometry_Documentation/differential_linear_logic.pdf`
//! - detected_kind: `pdf`
//! - extracted_at_utc: `2026-01-02T17:30:59+00:00`
//! - pages: `74`
//! - title: ``
//! - author: ``
//! - subject: ``
//! - keywords: ``
//! - creation_date: `Mon Jun  6 18:04:28 2016 PDT`
//! - mod_date: `Mon Jun  6 18:04:28 2016 PDT`
//! - encrypted: `no`
//!
//! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~text
//! arXiv:1606.01642v1 [cs.LO] 6 Jun 2016
//!
//! An introduction to Differential Linear Logic:
//! proof-nets, models and antiderivatives
//! Thomas Ehrhard
//! CNRS, IRIF, UMR 8243
//! Univ Paris Diderot, Sorbonne Paris Cité F-75205 Paris, France
//! February 2016
//!
//! Abstract
//! Differential Linear Logic enriches Linear Logic with additional logical
//! rules for the exponential connectives, dual to the usual rules of dereliction, weakening and contraction. We present a proof-net syntax for Differential Linear Logic and a categorical axiomatization of its denotational
//! models. We also introduce a simple categorical condition on these models
//! under which a general antiderivative operation becomes available. Last
//! we briefly describe the model of sets and relations and give a more detailed account of the model of finiteness spaces and linear and continuous
//! functions.
//!
//! Introduction
//! Extending Linear Logic (LL) with differential constructs has been considered by
//! Girard at a very early stage of the design of this system. This option appears
//! at various places in the conclusion of [Gir86], entitled Two years of linear logic:
//! selection from the garbage collector. In Section V.2 The quantitative attempt of
//! that conclusion, the idea of a syntactic Taylor expansion is explicitly mentioned
//! as a syntactic counterpart of the quantitative semantics of the λ-calculus [Gir88].
//! However it is contemplated there as a reduction process rather than as a transformation on terms. In Section V.5 The exponentials, the idea of reducing
//! λ-calculus substitution to a more elementary linear operation explicitly viewed
//! as differentiation is presented as one of the basic intuitions behind the exponential of LL. The connection of this idea with Krivine’s Machine [Kri85, Kri07]
//! and its linear head reduction mechanism [DR99] is explicitly mentioned. In this
//! mechanism, first considered by De Bruijn and called mini-reduction in [DB87],
//! it is only the head occurrence of a variable which is substituted during reduction. This restriction is very meaningful in LL: the head occurrence is the only
//! occurrence of a variable in a term which is linear.
//!
//! 1
//!
//!
//! --- PAGE BREAK ---
//! LL is based on the distinction of particular proofs among all proofs, that
//! are linear wrt. their hypotheses. The word linear has here two deeply related
//! meanings.
//! • An algebraic meaning: a linear morphism is a function which preserves
//! sums, linear combinations, joins, unions (depending on the context). In
//! most denotational models of LL, linear proofs are interpreted as functions
//! which are linear in that sense.
//! • An operational meaning: a proof is linear wrt. an hypothesis if the corresponding argument is used exactly once (neither erased nor duplicated)
//! during cut-elimination.
//! LL has an essential operation, called dereliction, which allows one to turn a
//! linear proof into a non linear one, or, more precisely, to forget the linearity of a
//! proof. Differentiation, which in some sense is the converse of dereliction, since
//! it turns a non linear morphism (proof) into a linear one, has not been included
//! in LL at an early stage of its development.
//! We think that there are two deep reasons for that omission.
//! • First, differentiation seems fundamentally incompatible with totality, a
//! denotational analogue of normalization usually considered as an essential
//! feature of any reasonable logical system. Indeed, turning a non-linear
//! proof into a linear one necessarily leads to a loss of information and to the
//! production of partial linear proofs. This is typically what happens when
//! one takes the derivative of a constant proof, which must produce a zero
//! proof.
//! • Second, they seem incompatible with determinism because, when one linearizes a proof obtained by contracting two linear inputs of a proof, one
//! has to choose between these two inputs, and there is no canonical way of
//! doing so: we take the non-deterministic superposition of the two possibilities. Syntactically, this means that one must accept the possibility of
//! adding proofs of the same formula, which is standard in mathematics, but
//! hard to accept as a primitive logical operation on proofs (although it is
//! present, in a tamed version, in the additive rules of LL).
//! The lack of totality is compatible with most mathematical interpretations of
//! proofs and with most denotational models of LL: Scott domains (or more precisely, prime algebraic complete lattices, see [Hut93, Win04, Ehr12]), dI-domains,
//! concrete data structures, coherence spaces, games, hypercoherence spaces etc.
//! Moreover, computer scientists are acquainted with the use of syntactic partial
//! objects (fix-point operators in programming languages, Böhm trees of the λcalculus etc.) and various modern proof formalisms, such as Girard’s Ludics,
//! also incorporate partiality for enlarging the world of “proof-objects” so as to
//! allow the simultaneous existence of “proofs” and “counter-proofs” in order to
//! obtain a rich duality theory on top of which a notion of totality discriminating
//! genuine proofs from partial proof-objects can be developed.
//! 2
//!
//!
//! --- PAGE BREAK ---
//! It is only when we observed that the differential extension of LL is the mirror
//! image of the structural (and dereliction) rules of LL that we considered this extension as logically meaningful and worth being studied more deeply. The price
//! to pay was the necessity of accepting an intrinsic non determinism and partiality in logic (these two extensions being related: failure is the neutral element
//! of non-determinism), but the gain was a new viewpoint on the exponentials,
//! related to the Taylor Formula of calculus.
//! In LL, the exponential is usually thought of as the modality of duplicable
//! information. Linear functions are not allowed to copy their arguments and are
//! therefore very limited in terms of computational expressive power, the exponential allows one to define non linear functions which can duplicate and erase
//! their arguments and are therefore much more powerful. This duplication and
//! erasure capability seems to be due to the presence of the rules of contraction
//! and weakening in LL, but this is not quite true: the genuinely infinite rule of LL
//! is promotion which makes a proof duplicable an arbitrary number of times, and
//! erasable. This fact could not be observed in LL because promotion is the only
//! rule of LL which allows one to introduce the “!” modality: without promotion,
//! it is impossible to build a proof object that can be cut on a contraction or a
//! weakening rule.
//! In Differential LL (DiLL), there are two new rules to introduce the “!” modality: coweakening and codereliction. The first of these rules allows one to introduce an empty proof of type !A and the second one allows one to turn a proof
//! of type A into a proof of type !A, without making it duplicable in sharp contrast
//! with the promotion rule. The last new rule, called cocontraction, allows one to
//! merge two proofs of type !A for creating a new proof of type !A. This latter
//! rule is similar to the tensor rule of ordinary LL with the difference that the two
//! proofs glued together by a cocontraction must have the same type and cannot
//! be distinguished anymore deterministically, whereas the two proofs glued by a
//! tensor can be separated again by cutting the resulting proof against a par rule.
//! These new rules are called costructural rules to stress the symmetry with the
//! usual structural rules of LL.
//! DiLL has therefore a finite fragment which contains the standard “?” rules
//! (weakening, contraction and dereliction) as well as the new “!” ones (coweakening, cocontraction and codereliction), but not the promotion rule. Cut elimination in this system generates sums of proofs, and therefore it is natural to endow
//! proofs with a vector space (or module) structure over a field (or more generally
//! over a semi-ring1 ). This fragment has the following pleasant properties:
//! • It enjoys strong normalization, even in the untyped case, as long as one
//! considers only proof-nets which satisfy a correctness criterion similar to
//! the standard Danos-Regnier criterion for multiplicative LL (MLL).
//! • In this fragment, all proofs are linear combinations of “simple proofs”
//! which do not contain linear combinations: this is possible because all the
//! 1 This general setting allows us to cover also “qualitative” situations where sums of proofs
//! are lubs in a poset.
//!
//! 3
//!
//!
//! --- PAGE BREAK ---
//! syntactic constructions of this fragment are multilinear. So proofs are
//! similar to polynomials or power series, simple proofs playing the role of
//! monomials in this algebraic analogy which is strongly suggested by the
//! denotational models of DiLL.
//! Moreover, it is possible to transform any instance of the promotion rule (which is
//! applied to a sub-proof π) into an infinite linear combination of proofs containing
//! copies of π: this is the Taylor expansion of promotion. This operation can be
//! applied hereditarily to all instances of the promotion rule in a proof, giving
//! rise to an infinite linear combinations of promotion-free DiLL simple proofs with
//! positive rational coefficients.
//! Outline.
//! We start with a syntactic presentation of DiLL, in a proof-net formalism which
//! uses terms instead of graphs (in the spirit of Abramsky’s linear chemical abstract
//! machine [Abr93] or of the formalisms studied by Fernandez and Mackie, see for
//! instance [FM99, MS08]) and we present a categorical formalism which allows
//! us to describe denotational models of DiLL. We define the interpretation of
//! proof-nets in such categories.
//! Then we shortly describe a differential λ-calculus formalism and we summarize some results, giving bibliographical references.
//! The end of the paper is devoted to concrete models of DiLL. We briefly review
//! the relational model, which is based on the ∗-autonomous category of sets and
//! relations (with the usual cartesian product of sets as tensor product) because
//! it underlies most denotational models of (differential) LL. Then we describe the
//! finiteness space model which was one of our main motivations for introducing
//! DiLL. We provide a thorough description of this model, insisting on various
//! aspects which were not covered by our initial presentation in [Ehr05] such as
//! linear boundedness (whose relevance in this semantical setting has been pointed
//! out by Tasson in [Tas09b, Tas09a]), or the fact that function spaces in the Kleisli
//! category admit an intrinsic description.
//! One important step in our presentation of the categorical setting for interpreting differential LL is the notion of an exponential structure. It is the
//! categorical counterpart of the finitary fragment of DiLL, that is, the fragment
//! DiLL0 where the promotion rule is not required to hold.
//! An exponential structure consists of a preadditive2 ∗-autonomous category
//! L together with an operation which maps any object X of L to an object
//! !X of L equipped with a structure of ⊗-bialgebra (representing the structural
//! and costructural rules) as well as a “dereliction” morphism in L(!X, X) and
//! a “codereliction” morphism L(X, !X). The important point here is that the
//! operation X 7→ !X is not assumed to be functorial (it has nevertheless to be a
//! 2 This means that the monoidal category is enriched over commutative monoids. Actually,
//! we assume more generally that it is enriched over the category of k-modules, where k is a
//! given semi-ring.
//!
//! 4
//!
//!
//! --- PAGE BREAK ---
//! functor on isomorphisms). Using this simple structure, we define in particular
//! morphisms ∂ X ∈ L(!X ⊗ X, !X) and ∂X ∈ L(!X, !X ⊗ X).
//! An element of L(!X, Y ) can be considered as a non-linear morphism from X
//! to Y (some kind of generalized polynomial, or analytical function), but these
//! morphisms cannot be composed. It is nevertheless possible to define a notion of
//! polynomial such morphism, and these polynomial morphisms can be composed,
//! giving rise to a category which is cartesian if L is cartesian.
//! By composition with ∂ X ∈ L(!X ⊗ X, !X), any element f of L(!X, Y ) can be
//! differentiated, giving rise to an element f 0 of L(!X ⊗X, Y ) that we consider as its
//! derivative3 . This operation can be performed again, giving rise to f 00 ∈ L(!X ⊗
//! X ⊗ X, Y ) and, assuming that cocontraction is commutative, this morphism is
//! symmetric in its two last linear parameters (a property usually known as Shwarz
//! Lemma).
//! In this general context, a very natural question arises. Given a morphism
//! g ∈ L(!X ⊗ X, Y ) whose derivative g 0 ∈ L(!X ⊗ X ⊗ X, Y ) is symmetric, can
//! one always find a morphism f ∈ L(!X, Y ) such that g = f 0 ? Inspired by the
//! usual proof of Poincaré’s Lemma, we show that such an antiderivative is always
//! available as soon as the natural morphism Id!X +(∂ X ∂X ) ∈ L(!X, !X) is an
//! isomorphism for each object X. We explain how this property is related to
//! a particular case of integration by parts. We also describe briefly a syntactic
//! version of antiderivatives in a promotion-free differential λ-calculus.
//! To interpret the whole of DiLL, including the promotion rule, one has to
//! assume that ! is an endofunctor on L and that this functor is endowed with a
//! structure of comonad and a monoidal structure; all these data have to satisfy
//! some coherence conditions wrt. the exponential structure. These conditions
//! are essential to prove that the interpretation of proof-nets is invariant under
//! the various reduction rules, among which the most complicated one is an LL
//! version of the usual chain rule of calculus. Our main references here are the
//! work of Bierman [Bie95], Melliès [Mel09] and, for the commutations involving
//! costructural logical rules, our concrete models [Ehr05, Ehr02], the categorical
//! setting developed by Blute, Cockett and Seely [BCS06] and, very importantly,
//! the work of Fiore [Fio07].
//! One major a priori methodological principle applied in this paper is to stick
//! to Classical Linear Logic, meaning in particular that the categorical models we
//! consider are ∗-autonomous categories. This is justified by the fact that most
//! of the concrete models we have considered so far satisfy this hypothesis (with
//! the noticeable exception of [BET12]) and it is only in this setting that the new
//! symmetries introduced by the differential and costructural rules appear clearly.
//! A lot of material presented in this paper could probably be carried to a more
//! general intuitionistic Linear Logic setting.
//! Some aspects of DiLL are only alluded to in this presentation, the most
//! by monoidal closedness, f 0 can be seen as an element of
//! L(!X, X ( Y ) where X ( Y is the object of morphisms from X to Y in L, that is, of linear
//! morphisms from X to Y , and the operation f 7→ f 0 satisfies all the ordinary properties of
//! differentiation.
//! 3 Or differential, or Jacobian:
//!
//! 5
//!
//!
//! --- PAGE BREAK ---
//! significant one being certainly the Taylor expansion formula and its connection
//! with linear head reduction. On this topic, we refer to [ER08, ER06, Ehr10].
//! Note on this version.
//! A final version of this paper will appear in Mathematical Structures in Computer
//! Science. The first version of this survey (already containing the material on
//! antiderivatives) has been written in 2011 and has been improved and enriched
//! since that time.
//!
//! Notations
//! In this paper, a set of coefficients is needed, which has to be a commutative
//! semi-ring. This set will be denoted as k. In Section 4.3, k will be assumed to
//! be a field but this assumption is not needed before that section.
//!
//! 1
//!
//! Syntax for DiLL proof-structures
//!
//! We adopt a presentation of proof-structures and proof-nets which is based on
//! terms and not on graphs. We believe that this presentation is more suitable
//! to formalizable mathematical developments, although it sometimes gives rise
//! to heavy notations, especially when one has to deal with the promotion rule
//! (Section 1.5). We try to provide graphical intuitions on proof-structures by
//! means of figures.
//!
//! 1.1
//!
//! General constructions
//!
//! 1.1.1
//!
//! Simple proof-structures.
//!
//! Let V be an infinite countable set of variables. This set is equipped with an
//! involution x 7→ x such that x 6= x for each x ∈ V.
//! Let u ⊆ V. An element x of u is bound in u if x ∈ u. One says that u is
//! closed if all the elements of u are bound in u. If x is not bound in u, one says
//! that x is free in u.
//! Let Σ be a set of tree constructors, given together with an arity map ar :
//! Σ → N.
//! Proof trees are defined as follows, together with their associated set of variables:
//! • if x ∈ V then x is a tree and V(x) = {x};
//! • if ϕ ∈ Σn (that is ϕ ∈ Σ and ar(ϕ) = n) and if t1 , . . . , tn are trees
//! with V(ti ) ∩ V(tj ) = ∅ for i 6= j, then t = ϕ(t1 , . . . , tn ) is a tree with
//! V(t) = V(ti ) ∪ · · · ∪ V(tn ). As usual, when ϕ is binary, we often use the
//! infix notation t1 ϕ t2 rather than ϕ(t1 , t2 ).
//!
//! 6
//!
//!
//! --- PAGE BREAK ---
//! A cut is an expression ht | t0 i where t and t0 are trees such that V(t)∩V(t0 ) = ∅.
//! We set V(c) = V(t) ∪ V(t0 ).
//! −
//! →
//! −
//! −c ; →
//! A simple proof-structure is a pair p = (→
//! t ) where t is a list of proof
//! →
//! −
//! trees and c is a list of cuts, whose sets of variables are pairwise disjoint.
//! −c does not matter; we could have used
//! Remark : The order of the elements of →
//! multisets instead of sequences. In the sequel, we consider these sequences of
//! cuts up to permutation.
//! Bound variables of V(p) can be renamed in the obvious way in p (rename
//! simultaneously x and x avoiding clashes with other variables which occur in
//! p) and simple proof-structures are considered up to such renamings: this is αconversion. Let FV(p) be the set of free variables of p. We say p is closed if
//! FV(p) = ∅.
//! The simplest simple proof-structure is of course ( ; ). A less trivial closed
//! simple proof-structure is (hx | xi ; ) which is a loop.
//! 1.1.2
//!
//! LL types.
//!
//! Let A be a set of type atoms ranged over by α, β, . . . , together with an involution
//! α 7→ α such that α 6= α. Types are defined as follows.
//! • if α ∈ A then α is a type;
//! • if A and B are types then A ⊗ B and A`B are types;
//! • if A is a type then !A and ?A are types.
//! The linear negation A⊥ of a type A is given by the following inductive definition:
//! α⊥ = α, (A ⊗ B)⊥ = A⊥ `B ⊥ ; (A`B)⊥ = A⊥ ⊗ B ⊥ ; (!A)⊥ = ?A⊥ and
//! (?A)⊥ = !A⊥ .
//! An MLL type is a type built using only the ⊗ and ` constructions4 .
//!
//! 1.2
//!
//! Proof-structures for MLL
//!
//! Assume that Σ2 = {⊗, `} and that ar(⊗) = ar(`) = 2.
//! A typing context is a finite partial function Φ (of domain D(Φ)) from V to
//! formulas such that Φ(x) = (Φ(x))⊥ whenever x, x ∈ D(Φ).
//! 1.2.1
//!
//! Typing rules.
//!
//! We first explain how to type MLL proof trees. The corresponding typing judgments of the form Φ `0 t : A where Φ is a typing context, t is a proof tree and
//! A is a formula.
//! The rules are
//! Φ, x : A `0 x : A
//! 4 We do not consider the multiplicative constants 1 and ⊥ because they are not essential
//! for our purpose.
//!
//! 7
//!
//!
//! --- PAGE BREAK ---
//! Φ `0 s : A
//! Φ `0 t : B
//! Φ `0 s ⊗ t : A ⊗ B
//!
//! Φ `0 s : A
//! Φ `0 t : B
//! Φ `0 s`t : A`B
//!
//! Given a cut c = hs | s0 i and a typing context Φ, one writes Φ `0 c if there is
//! a type A such that Φ `0 s : A and Φ `0 s0 : A⊥ .
//! Last, given a simple proof-structure p = (~c ; ~s) with ~s = (s1 , . . . , sn ) and
//! ~c = (c1 , . . . , ck ), a sequence Γ = (A1 , . . . , Al ) of formulas and a typing context
//! Φ, one writes Φ `0 p : Γ if l = n and Φ `0 si : Ai for 1 ≤ i ≤ n and Φ `0 ci for
//! 1 ≤ i ≤ k.
//! 1.2.2
//!
//! Logical judgments.
//!
//! A logical judgment is an expression Φ ` p : Γ where Φ is a typing context, p is
//! a simple proof-structure and Γ is a list of formulas.
//! If one can infer that Φ ` p : Γ, this means that the proof-structure p represents a proof of Γ. Observe that the inference rules coincide with the rules of
//! the MLL sequent calculus.
//! We give now these logical rules.
//! Φ, x : A, x : A⊥ ` ( ; x, x) : A, A⊥
//! Φ ` (~c ; t1 , . . . , tn ) : A1 , . . . , An
//! Φ ` (~c ; tσ(1) , . . . , tσ(n) ) : Aσ(1) , . . . , Aσ(n)
//!
//! axiom
//!
//! permutation rule, σ ∈ Sn
//!
//! →
//! − →
//! −
//! −c ; →
//! −
//! Φ ` (→
//! s , s) : Γ, A
//! Φ ` ( d ; t , t) : ∆, A⊥
//! −
//! →
//! −
//! −c , →
//! −
//! Φ ` (→
//! d , hs | ti ; →
//! s , t ) : Γ, ∆
//! −
//! −c ; →
//! Φ ` (→
//! t , s, t) : Γ, A, B
//! →
//! −
//! →
//! −
//! Φ ` ( c ; t , s`t) : Γ, A`B
//!
//! cut rule
//!
//! `-rule
//!
//! →
//! − →
//! −
//! −c ; →
//! −
//! Φ ` (→
//! s , s) : Γ, A
//! Φ ` ( d ; t , t) : ∆, B
//! − →
//! →
//! −
//! −c , →
//! Φ ` (→
//! d ;−
//! s , t , s ⊗ t) : Γ, ∆, A ⊗ B
//!
//! ⊗-rule
//!
//! We add the mix rule for completeness because it is quite natural denotationally.
//! Notice however that it is not necessary. In particular, mix-free proof-nets are
//! closed under cut elimination.
//! Φ ` (d~ ; ~t) : ∆
//! Φ ` (~c, d~ ; ~s, ~t) : Γ, ∆
//!
//! Φ ` (~c ; ~s) : Γ
//!
//! mix rule
//!
//! Lemma 1 If Φ ` p : Γ then Φ `0 p : Γ and V(p) is closed.
//! Proof.
//! Straightforward induction on derivations.
//!
//! 8
//!
//! 2
//!
//!
//! --- PAGE BREAK ---
//! 1.3
//!
//! Reducing proof-structures
//!
//! The basic reductions concern cuts, and are of the form
//! →
//! − →
//! −
//! c ;cut ( d ; t )
//! →
//! −
//! →
//! −
//! where c is a cut, d = (d1 , . . . , dn ) is a sequence of cuts and t = (t1 , . . . , tk ) is
//! a sequence fo trees.
//! With similar notational conventions, here are the deduction rules for the
//! reduction of MLL proof-structures.
//! →
//! − →
//! −
//! c ;cut ( d ; t )
//! context
//! →
//! − −
//! →
//! − →
//! − − →
//! −
//! (c, b ; →
//! s ) ;cut ( d , b ; →
//! s, t)
//! x∈
//! / V(s)
//! →
//! −
//! −
//! →
//! −
//! −c ; →
//! (hx | si, c ; t ) ;cut (→
//! t ) [s/x]
//!
//! ax-cut
//!
//! For applying the latter rule (see Figure 3), we assume that x ∈
//! / V(s). Without
//! this restriction, we would reduce the cyclic proof-structure (hx | xi ; ) to ( ; ) and
//! erase the cycle which is certainly not acceptable from a semantic viewpoint. For
//! instance, in a model of proof-structures based on finite dimension vector spaces,
//! the semantics of (hx | xi ; ) would be the dimension of the space interpreting the
//! type of x (trace of the identity).
//! Remark : We provide some pictures to help understand the reduction rules
//! on proof structures. In these pictures, logical proof-net constructors (such as
//! tensor, par etc.) are represented as white triangles labeled by the corresponding
//! symbol – they correspond to the cells of interaction nets or to the links of proofnets – and subtrees are represented as gray triangles.
//! Wires represent the edges of a proof tree. We also represent axioms and
//! cuts as wires: an axiom looks like
//! and a cut looks like
//! . In Figure 3, we
//! indicate the variables associated with the axiom, but in the next pictures, this
//! information will be kept implicit.
//! Figure 1 represents the simple proof-structure
//! p = (hs1 | s01 i, . . . , hsn | s0n i ; t1 , . . . , tk ) .
//! with free variables x1 , . . . , xl . The box named axiom links contains axioms connecting variables occurring in the trees s1 , . . . , sn , s01 , . . . , s0n , t1 , . . . , tk . When
//! we do not want to be specific about its content, we represent such a simple
//! proof-structure as in Figure 2 by a gray box with indices 1, . . . , k on its border
//! for locating the roots of the trees of p. The same kind of notation will be used
//! also for proof-structures which are not necessarily simple, see the beginning of
//! Paragraph 1.4.3 for this notion.
//! In MLL, we have only one basic reduction (see Figure 4):
//! hs1 `s2 | t1 ⊗ t2 i ;cut (hs1 | t1 i, hs2 | t2 i ; )
//! 9
//!
//!
//! --- PAGE BREAK ---
//! x1
//!
//! p=
//!
//! ...
//! s1
//!
//! ...
//! s01
//!
//! ...
//!
//! xl
//! ...
//! axiom links
//! ...
//! ...
//! sn
//! s0n
//!
//! ...
//! t1
//!
//! ...
//!
//! ...
//! tk
//!
//! Figure 1: A simple proof-structure
//! x1
//!
//! xl
//! ...
//! p
//!
//! 1
//!
//! ...
//!
//! k
//!
//! Figure 2: A synthetic representation of the proof-structure of Figure 1
//!
//! 1.4
//!
//! DiLL0
//!
//! This is the promotion-free fragment of differential LL. In DiLL0 , one extends the
//! signature of MLL with new constructors:
//! • Σ0 = {w, w}, called respectively weakening and coweakening.
//! • Σ1 = {d, d}, called respectively dereliction and codereliction.
//! • Σ2 = {`, ⊗, c, c}, the two new constructors being called respectively contraction and cocontraction.
//! • Σn = ∅ for n > 2.
//! 1.4.1
//!
//! Typing rules.
//!
//! The typing rules for the four first constructors are similar to those of MLL.
//! Φ `0 w : ?A
//!
//! Φ `0 w : !A
//!
//! Φ `0 t : A
//! Φ `0 d(t) : ?A
//!
//! Φ `0 t : A
//! Φ `0 d(t) : !A
//!
//! The two last rules require the subtrees to have the same type.
//! Φ `0 s1 : ?A
//! Φ `0 s2 : ?A
//! Φ `0 c(s1 , s2 ) : ?A
//!
//! Φ `0 s1 : !A
//! Φ `0 s2 : !A
//! Φ `0 c(s1 , s2 ) : !A
//!
//! 10
//!
//!
//! --- PAGE BREAK ---
//! s
//! x
//!
//! x
//! −
//! −c ; →
//! (→
//! t)
//!
//! s
//!
//! ;cut
//!
//! −
//! −c ; →
//! (→
//! t)
//!
//! Figure 3: The axiom/cut reduction
//! s1
//!
//! s2
//!
//! t1
//!
//! t2
//! s1
//!
//! `
//!
//! ⊗
//!
//! s2
//!
//! t1
//!
//! t2
//!
//! ;cut
//!
//! Figure 4: The tensor/par reduction
//! 1.4.2
//!
//! Logical rules.
//!
//! The additional logical rules are as follows.
//! −c ; →
//! −
//! Φ ` (→
//! s):Γ
//! weakening
//! →
//! −
//! →
//! −
//! Φ ` ( c ; s , w) : Γ, ?A
//!
//! −c ; →
//! −
//! Φ ` (→
//! s , s) : Γ, A
//! →
//! −
//! →
//! −
//! Φ ` ( c ; s , d(s)) : Γ, ?A
//!
//! dereliction
//!
//! −c ; →
//! −
//! Φ ` (→
//! s , s) : Γ, A
//! →
//! −
//! →
//! −
//! Φ ` ( c ; s , d(s)) : Γ, !A
//!
//! co-dereliction
//!
//! −c ; →
//! −
//! Φ ` (→
//! s , s1 , s2 ) : Γ, ?A, ?A
//! →
//! −
//! →
//! −
//! Φ ` ( c ; s , c(s1 , s2 )) : Γ, ?A
//!
//! contraction
//!
//! →
//! − →
//! −
//! −c ; →
//! −
//! Φ ` (→
//! s , s) : Γ, !A
//! Φ ` ( d ; t , t) : ∆, !A
//! − →
//! →
//! −
//! −c , →
//! Φ ` (→
//! d ;−
//! s , t , c(s, t)) : Γ, ∆, !A
//! 1.4.3
//!
//! co-weakening
//!
//! Φ ` ( ; w) : !A
//!
//! co-contraction
//!
//! Reduction rules.
//!
//! To describe the reduction rules associated with these new constructions, we need
//! to introduce formal sums (or more generally k-linear combinations) of simple
//! proof-structures called proof-structures in the sequel, and denoted with capital
//! letters P, Q, . . . Such an extension by linearity of the syntax was already present
//! in [ER03].
//! The empty linear combination 0 is a particular proof-structure which plays
//! an important role. As linear combinations, proof-structures can be linearly
//! combined.
//! The typing rule for linear combinations is
//! 11
//!
//!
//! --- PAGE BREAK ---
//! ?
//!
//! !
//!
//! ;cut
//!
//! Figure 5: Weakening/coweakening reduction
//! s
//! ?
//!
//! t
//! ;cut 0
//!
//! !
//!
//! ?
//!
//! !
//!
//! ;cut 0
//!
//! Figure 6: Dereliction/coweakening and weakening/codereliction reductions
//! ∀i ∈ {1, . . . , n} Φ ` pi : Γ and µi ∈ k
//! Pn
//! Φ ` i=1 µi pi : Γ
//!
//! sum
//!
//! The new basic reduction rules are:
//! hw | wi ;cut ( ; )
//!
//! see Figure 5.
//!
//! hd(s) | wi ;cut 0
//! hw | d(t)i ;cut 0
//!
//! see Figure 6.
//!
//! hc(s1 , s2 ) | wi ;cut (hs1 | wi, hs2 | wi ; )
//! hw | c(t1 , t2 )i ;cut (hw | t1 i, hw | t2 i ; )
//! hd(s) | d(t)i ;cut (hs | ti ; )
//!
//! see Figure 7.
//!
//! see Figure 8.
//!
//! hc(s1 , s2 ) | d(t)i ;cut (hs1 | d(t)i, hs2 | wi ; ) + (hs1 | wi, hs2 | d(t)i ; )
//! hd(s) | c(t1 , t2 )i ;cut (hd(s) | t1 i, hw | t2 i ; ) + (hw | t1 i, hd(s) | t2 i ; )
//! see Figure 9.
//! hc(s1 , s2 ) | c(t1 , t2 )i
//! ;cut (hs1 | c(x11 , x12 )i, hs2 | c(x21 , x22 )i, hc(x11 , x21 ) | t1 i, hc(x12 , x22 ) | t2 i ; )
//! see Figure 10.
//! In the last reduction rule, the four variables that we introduce are pairwise
//! distinct and fresh. Up to α-conversion, the choice of these variables is not
//! relevant.
//! The contextual rule must be extended, in order to take sums into account.
//! c ;cut P
//! P
//! →
//! − →
//! − →
//! −
//! −
//! →
//! − →
//! − →
//! →
//! − P ·( c , b ; s, t )
//! (c, b ; s ) ;cut p=(→
//! −
//! c ; t) p
//!
//! context
//!
//! Remark :
//! In the premise of this rule, P is a linear combination of proof−
//! −c ; →
//! structures, so that for a given proof-structure p = (→
//! t ), Pp ∈ k is the
//! coefficient of the proof-structure p in this linear combination P . The sum which
//! appears in the conclusion ranges over all possible proof-structures p, but there
//! are only finitely many p’s such that Pp 6= 0 so that this sum is actually finite.
//! →
//! − −
//! A particular case of this rule is c ;cut 0 ⇒ (c, b ; →
//! s ) ;cut 0.
//! 12
//!
//!
//! --- PAGE BREAK ---
//! s1
//!
//! ?
//!
//! s2
//! ?
//!
//! !
//!
//! t1
//!
//! t2
//! !
//!
//! s1
//!
//! ;cut
//!
//! ;cut
//!
//! ?
//!
//! s2
//!
//! !
//!
//! t1
//!
//! ?
//!
//! !
//!
//! t2
//!
//! Figure 7: Contraction/coweakening and weakening/cocontraction reductions
//! s
//!
//! t
//!
//! ?
//!
//! !
//!
//! s
//! ;cut
//!
//! t
//!
//! Figure 8: Dereliction/codereliction reduction
//!
//! 1.5
//!
//! Promotion
//!
//! −c ; →
//! −
//! Let p = (→
//! s ) be a simple proof-structure. The width of p is the number of
//! −
//! elements of the sequence →
//! s.
//! By definition, a proof-structure of width n is a finite linear combination of
//! simple proof-structures of width n.
//! Observe that 0 is a proof-structure of width n for all n.
//! Let P be a proof-structure5 of width n + 1. We introduce a new constructor6
//! called promotion box, of arity n:
//! P !(n) ∈ Σn .
//! The presence of n in the notation is useful only in the case where P = 0 so it
//! can most often be omitted. The use of a non necessarily simple proof structure
//! P in this construction is crucial: promotion is not a linear construction and is
//! actually the only non linear construction of (differential) LL.
//! So if t1 , . . . , tn are trees, P !(n) (t1 , . . . , tn ) is a tree. Pictorially, this tree will
//! typically be represented as in Figure 11. A simple net p appearing in P is
//! −c ; →
//! −
//! −
//! typically of the form (→
//! s ) and its width is n + 1, so that →
//! s = (s1 , . . . , sn , s).
//! The indices 1, . . . , n and • which appear on the gray rectangle representing P
//! stand for the roots of these trees s1 , . . . , sn and s.
//! 5 To be completely precise, we should also provide a typing environment for the free variables
//! of P ; this can be implemented by equipping each variable with a type.
//! 6 The definitions of the syntax of proof trees and of the signature Σ are mutually recursive
//! when promotion is taken into account.
//!
//! 13
//!
//!
//! --- PAGE BREAK ---
//! s1
//!
//! s2
//!
//! t
//!
//! ?
//!
//! s1
//!
//! t1
//!
//! s
//! ?
//!
//! s
//!
//! t1
//!
//! !
//!
//! s
//!
//! t1
//!
//! t2
//!
//! ?
//!
//! +
//!
//! ?
//!
//! t
//!
//! !
//!
//! t2
//! ?
//!
//! ;cut
//!
//! s2
//!
//! +
//!
//! !
//!
//! t2
//! !
//!
//! s1
//! !
//!
//! ;cut
//!
//! !
//!
//! s2
//!
//! t
//!
//! ?
//!
//! Figure 9: Contraction/codereliction and dereliction/cocontraction reductions
//! s1
//!
//! s2
//!
//! t1
//!
//! ?
//!
//! t2
//! ;cut
//!
//! !
//!
//! s1
//!
//! s2
//!
//! !
//!
//! !
//!
//! ?
//!
//! t1
//!
//! ?
//!
//! t2
//!
//! Figure 10: Contraction/cocontraction reduction
//! 1.5.1
//!
//! Typing rule.
//!
//! The typing rule for this construction is
//! ⊥
//! Φ `0 P : ?A⊥
//! 1 , . . . , ?An , B
//!
//! Φ `0 ti : !Ai
//!
//! !(n)
//!
//! (t1 , . . . , tn ) : !B
//!
//! Φ `0 P
//! 1.5.2
//!
//! (i = 1, . . . , n)
//!
//! Logical rule.
//!
//! The logical rule associated with this construction is the following.
//! →
//! −
//! −
//! ⊥
//! Φ ` P : ?A⊥
//! Φ ` (→
//! ci ; ti , ti ) : Γi , !Ai (i = 1, . . . , n)
//! 1 , . . . , ?An , B
//! →
//! −
//! →
//! −
//! −
//! −
//! Φ ` (→
//! c ,...,→
//! c ; t , . . . , t , P !(n) (t , . . . , t )) : Γ , . . . , Γ , !B
//! 1
//!
//! n
//!
//! 1
//!
//! n
//!
//! 1
//!
//! n
//!
//! 1
//!
//! n
//!
//! Remark : This promotion rule is of course highly debatable. We choose this
//! presentation because it is compatible with our tree-based presentation of proofstructures.
//!
//! 14
//!
//!
//! --- PAGE BREAK ---
//! t1
//!
//! tn
//! ···
//!
//! 1
//!
//! n
//!
//! P
//! •
//! !
//! Figure 11: Graphical representation of a tree whose outermost constructor is a
//! promotion box
//! t1
//!
//! tn
//! ···
//!
//! 1
//!
//! n
//!
//! ;cut
//!
//! P
//! •
//! !
//!
//! t1
//!
//! ?
//!
//! ···
//!
//! tn
//!
//! ?
//!
//! ?
//!
//! Figure 12: Promotion/weakening reduction
//! 1.5.3
//!
//! Cut elimination rules.
//!
//! The basic reductions associated with promotion are as follows.
//! hP !(n) (t1 , . . . , tn ) | wi ;cut (ht1 | wi, . . . , htn | wi ; )
//! hP !(n) (t1 , . . . , tn ) | d(s)i ;cut
//! X
//!
//! see Figure 12.
//!
//! −c , hs | t i, . . . , hs | t i, hs0 | si ; )
//! Pp · (→
//! 1 1
//! n n
//!
//! −
//! −
//! p=(→
//! c ;→
//! s ,s0 )
//!
//! see Figure 13.
//! hP
//!
//! !(n)
//!
//! (t1 , . . . , tn ) | c(s1 , s2 )i ;cut (hP
//!
//! !(n)
//!
//! (x1 , . . . , xn ) | s1 i, hP !(n) (y1 , . . . , yn ) | s2 i,
//!
//! ht1 | c(x1 , y1 )i, . . . , htn | c(xn , yn )i ; )
//! see Figure 14.
//! In the second reduction rule, one has to avoid clashes of variables.
//! In the last reduction rules, the variables x1 , . . . , xn and y1 , . . . , yn that we
//! introduce together with their covariables are assumed to be pairwise distinct
//! and fresh. Up to α-conversion, the choice of these variables is not relevant.
//! 1.5.4
//!
//! Commutative reductions.
//!
//! There are also auxiliary reduction rules sometimes called commutative reductions which do not deal with cuts — at least in the formalization of nets we
//! present here.
//! 15
//!
//!
//! --- PAGE BREAK ---
//! t1
//!
//! tn
//! ···
//!
//! n
//!
//! 1
//!
//! s
//!
//! P
//! •
//! !
//!
//! t1
//!
//! ;cut
//!
//! ···
//!
//! tn
//! n
//!
//! s
//!
//! P
//! 1 •
//! ···
//!
//! ?
//!
//! Figure 13: Promotion/dereliction reduction
//! t1
//!
//! tn
//! ···
//!
//! n
//!
//! 1
//!
//! s1
//!
//! P
//! •
//! !
//!
//! ;cut
//!
//! ?
//! ···
//!
//! s1
//!
//! s2
//!
//! 1
//!
//! n
//!
//! P
//! •
//!
//! t1
//!
//! ?
//!
//! ···
//!
//! tn
//!
//! ···
//! ?
//!
//! 1
//!
//! !
//!
//! P
//! •
//!
//! n
//!
//! s2
//!
//! !
//!
//! Figure 14: Promotion/contraction reduction
//! The format of these reductions is
//! t ;com P
//! where t is a simple tree and P is a (not necessarily simple) proof-structure whose
//! width is exactly 1.
//! The first of these reductions is illustrated in Figure 15 and deals with the
//! interaction between two promotions.
//! P !(n+1) (t1 , . . . , ti−1 , Q!(k) (ti , . . . , tk+i−1 ), tk+i , . . . , tk+n )
//! ;com ( ; R!(k+n) (t1 , . . . , tk+n ))
//! where
//! R=
//!
//! X
//!
//! −c , hs | Q!(k) (x , . . . , x )i ;
//! Pp · (→
//! i
//! 1
//! k
//!
//! −
//! p=(→
//! c ; s1 ,...,sn+1 ,s)
//!
//! s1 , . . . , si−1 , x1 , . . . , xk , si+1 , . . . , sn+1 , s) .
//!
//! 16
//!
//! (1)
//!
//!
//! --- PAGE BREAK ---
//! Remark : In Figure 15 and 17, for graphical reasons, we don’t follow exactly the
//! notations used in the text. For instance in Figure 15, the correspondence with
//! the notations of (1) is given by v1 = t1 ,. . . ,vi−1 = ti−1 , u1 = ti ,. . . , uk = tk+i−1 ,
//! vi = tk+i ,. . . ,vn = tk+n .
//! Remark : Figure 15 is actually slightly incorrect as the connections between the
//! “auxiliary ports” of the cocontraction rule within the promotion box of the right
//! hand proof-structure and the main ports of the trees u1 , . . . , uk are represented
//! as vertical lines whereas they involve axioms (corresponding to the pairs (xi , xi )
//! for i = 1, . . . , k in the formula above). The same kind of slight incorrectness
//! occurs in figure 17.
//! The three last commutative reductions deal with the interaction between a
//! promotion and the costructural rules.
//! Interaction between a promotion and a coweakening, see Figure 16:
//! P !(n+1) (t1 , . . . , ti−1 , w, ti , . . . , tn ) ;com ( ; R!(n) (t1 , . . . , tn ))
//! where
//! R=
//!
//! X
//!
//! −c , hs | wi ; s , . . . , s , s , . . . , s
//! Pp · (→
//! i
//! 1
//! i−1 i+1
//! n+1 , s) .
//!
//! −
//! −
//! p=(→
//! c ;→
//! s ,s)
//!
//! Interaction between a promotion and a cocontraction, see Figure 17:
//! P !(n+1) (t1 , . . . , ti−1 , c(ti , ti+1 ), ti+2 , . . . , tn+2 ) ;com ( ; R!(n+2) (t1 , . . . , tn+2 ))
//! (2)
//! where
//! X
//! −c , hs | c(x, y)i ; s , . . . , s , x, y, s , . . . , s , s) .
//! R=
//! Pp (→
//! i
//! 1
//! i−1
//! i
//! n
//! −
//! −
//! p=(→
//! c ;→
//! s ,s)
//!
//! The interaction between a promotion and a codereliction is a syntactic version of the chain rule of calculus, see Figure 18.
//! P !(n+1) (t1 , . . . , ti−1 , d(u), ti+1 , . . . , tn+1 )
//! X
//! ;com
//! Pp · (hsi | d(u)i,
//! −
//! p=(→
//! c ; s1 ,...,sn+1 ,s)
//!
//! hc(x1 , s1 ) | t1 i, . . . , hc(x\
//! i , si ) | ti i, . . . , hc(xn+1 , sn+1 ) | tn+1 i ;
//! c(P !(n+1) (x1 , . . . , xi−1 , w, xi+1 , . . . , xn+1 ), d(s)))
//! where we use the standard notation a1 , . . . , abi , . . . , an for the sequence
//! a1 , . . . , ai−1 , ai+1 , . . . , an .
//! We also have to explain how these commutative reductions can be used in
//! arbitrary contexts. We deal first with the case where such a reduction occurs
//! under a constructor symbol ϕ ∈ Σn+1 .
//! 17
//!
//!
//! --- PAGE BREAK ---
//! ···
//! 1
//!
//! ···
//! 1
//!
//! k
//!
//! Q
//! •
//!
//! v1
//!
//! vn
//!
//! ··· ···
//! i
//!
//! k
//!
//! Q
//! •
//!
//! v1
//! ;com
//!
//! !
//! 1
//!
//! uk
//!
//! u1
//!
//! uk
//!
//! u1
//!
//! vn
//!
//! !
//! ···
//!
//! n
//!
//! n
//!
//! 1
//!
//! P
//! •
//!
//! P
//! •
//!
//! !
//!
//! i
//!
//! !
//!
//! Figure 15: Promotion/promotion commutative reduction
//!
//! t1
//!
//! t1
//!
//! tn
//!
//! tn
//!
//! !
//! ··· ···
//! 1
//!
//! i
//!
//! !
//! n
//!
//! ;com
//!
//! P
//! •
//!
//! ···
//! 1
//!
//! P
//! •
//!
//! !
//!
//! n
//! i
//!
//! !
//! Figure 16: Promotion/coweakening commutative reduction
//! t ;com P
//! P
//! →
//! −
//! →
//! −
//! →
//! −
//! →
//! −
//! →
//! −
//! −
//! ϕ( u , t, v ) ;com p=(→
//! c ; w) Pp · ( c ; ϕ( u , w, v ))
//! Next we deal with the case where t occurs in outermost position in a proofstructure. There are actually two possibilities.
//! t ;com P
//! P
//! − →
//! →
//! −
//! →
//! −
//! →
//! −
//! −c , →
//! −
//! −
//! ( c ; u , t, v ) ;com p=(→
//! P · (→
//! d ;−
//! u , w, →
//! v)
//! d ; w) p
//! t ;com P
//! P
//! −
//! →
//! −
//! →
//! −
//! →
//! −
//! −c , →
//! −
//! (ht | t0 i, c ; t ) ;com p=(→
//! P · (→
//! d , hw | t0 i ; t )
//! d ; w) p
//! We use ; for the union of the reduction relations ;cut and ;com .
//! This formalization of nets enjoys a subject reduction property.
//! Theorem 2 If Φ ` p : Γ and p ; P then Φ0 ` P : Γ for some Φ0 which extends
//! Φ.
//! 18
//!
//!
//! --- PAGE BREAK ---
//! u1
//!
//! v1
//!
//! u2
//!
//! v1
//!
//! u1
//!
//! u2
//!
//! vn
//!
//! vn
//! !
//!
//! !
//!
//! ··· ···
//! 1
//!
//! i
//!
//! ···
//!
//! ;com
//!
//! n
//!
//! n
//!
//! 1
//!
//! P
//! •
//!
//! P
//! •
//!
//! !
//!
//! !
//!
//! i
//!
//! Figure 17: Promotion/cocontraction commutative reduction rules
//! u
//! u
//! t1
//!
//! P
//! i
//! n•
//! ··· ···
//!
//! !
//!
//! !
//!
//! tn
//!
//! 1
//!
//! !
//! ··· ···
//! ··· ···
//! 1
//!
//! i
//!
//! P
//! •
//!
//! n
//!
//! ;com
//!
//! 1
//!
//! i
//!
//! P
//! •
//!
//! n
//!
//! t1
//!
//! ?
//!
//! ···
//!
//! tn
//!
//! ?
//!
//! !
//!
//! !
//!
//! !
//! !
//! Figure 18: Promotion/codereliction commutative reduction (Chain Rule)
//! The proof is a rather long case analysis. We need to consider possible extensions
//! of Φ because of the fresh variables which are introduced by several reduction
//! rules.
//!
//! 1.6
//!
//! Correctness criterion and properties of the reduction
//!
//! Let P be proof-structure, Φ be a closed typing context and Γ be a sequence of
//! formulas such that Φ `0 P : Γ. One says that P is a proof-net if it satisfies
//! Φ ` P : Γ. A correctness criterion is a criterion on P which guarantees that P
//! is a proof-net; of course, saying that Φ ` P : Γ is a correctness criterion, but
//! is not a satisfactory one because it is not easy to prove that it is preserved by
//! reduction.
//! Various such criteria can be found in the literature, but most of them apply
//! to proof-structures considered as graphical objects and are not very suitable
//! to our term-based approach. We rediscovered recently a correctness criterion
//! initially due to Rétoré [Ret03] which seems more convenient for the kind of
//!
//! 19
//!
//!
//! --- PAGE BREAK ---
//! presentation of proof-structures that we use here, see [Ehr14]. This criterion,
//! which is presented for MLL, can easily be extended to the whole of DiLL.
//! So far, the reduction relation ; is defined as a relation between simple
//! proof-structures and proof-structures. It must be extended to a relation between
//! arbitrary proof-structures. This is done by means of the following rules
//! p;P
//! p+Q;P +Q
//!
//! p;P
//! µ ∈ k \ {0}
//! µ·p;µ·P
//!
//! As it is defined, our reduction relation does not allow us to perform the reduction
//! within boxes. To this end, one should add the following rule.
//! P ;Q
//! P !(n) (t1 , . . . , tn ) ; Q!(n) (t1 , . . . , tn )
//! It is then possible to prove basic properties such as confluence and normalization7 . For these topics, we refer mainly to the work of Pagani [Pag09], Tranquilli [PT09, Tra09, PT11], Gimenez [Gim11]. We also refer to Vaux [Vau09]
//! for the link between the algebraic properties of k and the properties of ;, in a
//! simpler λ-calculus setting.
//! Of course these proofs should be adapted to our presentation of proof structures. This has not been done yet but we are confident that it should not lead
//! to difficulties.
//!
//! 2
//!
//! Categorical denotational semantics
//!
//! We describe now the denotational semantics of DiLL in a general categorical
//! setting. This will give us an opportunity to provide more intuitions about
//! the rules of this system. More intuition about the meaning of the differential
//! constructs of DiLL is given in Section 3.
//!
//! 2.1
//!
//! Notations and conventions
//!
//! Let C be a category. Given objects X and Y of C, we use C(X, Y ) for the set
//! of morphisms from X to Y . Given f ∈ C(X, Y ) and g ∈ C(Y, Z), we use g f for
//! the composition of f and g, which belongs to C(X, Z). In specific situations, we
//! use also the notation g ◦ f . When there are no ambiguities, we use X instead
//! of IdX to denote the identity from X to X.
//! Given n ∈ N and a functor F : C → D, we use the same notation F
//! for the functor C n → Dn defined in the obvious manner: F (X1 , . . . , Xn ) =
//! (F (X1 ), . . . , F (Xn )) and similarly for morphisms. If F, G : C → D are functors
//! and if T is a natural transformation, we use again the same notation T for the
//! corresponding natural transformation between the functors F, G : C n → Dn , so
//! that TX1 ,...,Xn = (TX1 , . . . , TXn ).
//! 7 For confluence, one needs to introduce an equivalence relation on proof-structures which
//! expresses typically that contraction is associative, see [Tra09]. For normalization, some conditions have to be satisfied by k; typically, it holds if one assumes that k = N but difficulties
//! arise if k has additive inverses.
//!
//! 20
//!
//!
//! --- PAGE BREAK ---
//! 2.2
//!
//! Monoidal structure
//!
//! A symmetric monoidal category is a structure (L, I, , λ, ρ, α, σ) where L is a
//! category, I is an object of L,  : L2 → L is a functor and λX ∈ L(I  X, X),
//! ρX ∈ L(X  I, X), αX,Y,Z ∈ L((X  Y )  Z, X  (Y  Z)) and σX,Y ∈
//! L(X  Y, Y  X) are natural isomorphisms satisfying coherence conditions
//! which can be expressed as commutative diagrams, and that we do not recall
//! here. Following McLane [Mac71], we present these coherence conditions using
//! a notion of monoidal trees (called binary words in [Mac71]).
//! Monoidal trees (or simply trees when there are no ambiguities) are defined
//! by the following syntax.
//! • hi is the empty tree
//! • ∗ is the tree consisting of just one leaf
//! • and, given trees τ1 and τ2 , hτ1 , τ2 i is a tree.
//! Let L(τ ) be the number of leaves of τ , defined by
//! L(hi) = 0
//! L(∗) = 1
//! L(hτ1 , τ2 i) = L(τ1 ) + L(τ2 ) .
//! Let Tn be the set of trees τ such that L(τ ) = n. This set is infinite for all n.
//! Let τ ∈ Tn . Then we define in an obvious way a functor τ : Ln → L. On
//! object, it is defined as follows:
//! hi = I
//! ∗ X = X
//! →
//! −
//! →
//! −
//! hτ1 ,τ2 i (X1 , . . . , XL(τ1 ) , Y1 , . . . , YL(τ2 ) ) = (τ1 ( X ))  (τ2 ( Y )) .
//! The definition on morphisms is similar.
//! 2.2.1
//!
//! Generalized associativity.
//!
//! Given τ1 , τ2 ∈ Tn , the isomorphisms λ, ρ and α of the monoidal structure of L
//! allow us to build an unique natural isomorphism ττ12 from τ1 to τ2 . We have
//! in particular
//! hhi,∗i
//!
//! λX = ∗
//!
//! X
//!
//! h∗,hii
//! ρX = ∗
//! X
//! hh∗,∗i,∗i
//!
//! αX,Y,Z = h∗,h∗,∗ii
//!
//! X,Y,Z
//!
//! The coherence commutation diagrams (which include the McLane Pentagon)
//! allow one indeed to prove that all the possible definitions of an isomorphism
//!
//! 21
//!
//!
//! --- PAGE BREAK ---
//! →
//! −
//! →
//! −
//! τ1 ( X ) → τ2 ( X ) using these basic ingredients give rise to the same result.
//! This is McLane coherence Theorem for monoidal categories. In particular the
//! following properties will be quite useful:
//! − = Id τ →
//! −
//! ττ →
//! X
//!  (X )
//!
//! 1−
//! − τ
//! − .
//! and ττ23 →
//! = ττ13 →
//! τ2 →
//! X
//! X
//! X
//!
//! (3)
//!
//! →
//! −
//! We shall often omit the indexing sequence X when using these natural isomor−.
//! phisms, writing στ instead of στ →
//! X
//! 2.2.2
//!
//! Generalized symmetry.
//!
//! Let n ∈ N. Let ϕ ∈ Sn , we define a functor ϕ
//! b : Ln → Ln by ϕ(X
//! b 1 , . . . , Xn ) =
//! (Xϕ(1) , . . . , Xϕ(n) )
//! Assume that the monoidal category L is also symmetric. The corresponding
//! b ϕ,τ from the
//! additional structure allows one to define a natural isomorphism 
//! τ
//! τ
//! ϕ,τ
//! b
//! functor  to the functor  ◦ ϕ.
//! b The correspondence ϕ 7→ 
//! is of course
//! functorial. Moreover, given σ, τ ∈ Tn and ϕ ∈ Sn , the following diagram is
//! commutative
//! b ϕ,σ
//! →
//! −
//! →
//! − 
//! σ ϕ(
//! b X)
//! σ X
//! σ
//! τ
//!
//! →
//! −
//! τ X
//!
//! σ
//! τ
//! b ϕ,τ
//! 
//!
//! →
//! −
//! τ ϕ(
//! b X)
//!
//! (4)
//!
//! This is a consequence of McLane coherence Theorem for symmetric monoidal
//! categories.
//!
//! 2.3
//!
//! *-autonomous categories
//!
//! A *-autonomous category is a symmetric monoidal category (L, ⊗, λ, ρ, α, σ)
//! equipped with the following structure:
//! • an endomap on the objects of L that we denote as X 7→ X ⊥ ;
//! • for each object X, an evaluation morphism ev⊥ ∈ L(X ⊥ ⊗ X, ⊥), where
//! ⊥ = 1⊥ ;
//! • a curryfication function cur⊥ : L(U ⊗ X, ⊥) → L(U, X ⊥ )
//! subject to the following equations (with f ∈ L(U ⊗ X, ⊥) and g ∈ L(V, U ), so
//! that g ⊗ X ∈ L(V ⊗ X, U ⊗ X)):
//! ev⊥ (cur⊥ (f ) ⊗ X) = f
//! cur⊥ (f ) g = cur⊥ (f (g ⊗ X))
//! cur⊥ (ev⊥ ) = Id .
//! Then cur⊥ is a bijection. Indeed, let g ∈ L(U, X ⊥ ). Then g ⊗ X ∈ L(U ⊗
//! X, X ⊥ ⊗ X) and hence ev⊥ (g ⊗ X) ∈ L(U ⊗ X, ⊥). The equations allow one to
//! prove that the function g 7→ ev⊥ (g ⊗ X) is the inverse of the function cur⊥ .
//! 22
//!
//!
//! --- PAGE BREAK ---
//! For any object X of L, let ηX = cur⊥ (ev⊥ σX,X ⊥ ) ∈ L(X, X ⊥⊥ ).
//! The operation X 7→ X ⊥ can be extended into a functor Lop → L as follows.
//! Let f ∈ L(X, Y ), then ηY f ∈ L(X, Y ⊥⊥ ), so ev⊥ ((ηY f )⊗Y ⊥ ) ∈ L(X ⊗Y ⊥ , ⊥)
//! and we set f ⊥ = cur⊥ (ev⊥ ((ηY f ) ⊗ Y ⊥ ) σY ⊥ ,X ) ∈ L(Y ⊥ , X ⊥ ). It can be
//! checked that this operation is functorial.
//! We assume last that ηX is an iso for each object X.
//! One sets X ( Y = (X ⊗ Y ⊥ )⊥ and one defines an evaluation morphism
//! ev ∈ L((X ( Y ) ⊗ X, Y ) as follows. We have
//! ev⊥ ∈ L((X ⊗ Y ⊥ )⊥ ⊗ (X ⊗ Y ⊥ ), ⊥)
//! hence
//!
//! hh∗,∗i,∗i
//!
//! ev⊥ ⊗h∗,h∗,∗ii ∈ L(((X ⊗ Y ⊥ )⊥ ⊗ X) ⊗ Y ⊥ , ⊥) ,
//! therefore
//!
//! hh∗,∗i,∗i
//!
//! cur⊥ (ev⊥ ⊗h∗,h∗,∗ii ) ∈ L((X ⊗ Y ⊥ )⊥ ⊗ X, Y ⊥⊥ )
//! and we set
//! hh∗,∗i,∗i
//!
//! ev = η −1 cur⊥ (ev⊥ ⊗h∗,h∗,∗ii ) ∈ L((X ⊗ Y ⊥ )⊥ ⊗ X, Y ).
//! Let f ∈ L(U ⊗ X, Y ). We have η f ∈ L(U ⊗ X, Y ⊥⊥ ), hence cur⊥ −1 (η f ) ∈
//! h∗,h∗,∗ii
//! L((U ⊗ X) ⊗ Y ⊥ , ⊥), so cur⊥ −1 (η f ) ⊗hh∗,∗i,∗i ∈ L(U ⊗ (X ⊗ Y ⊥ ), ⊥) and we
//! can define a linear curryfication of f as
//! h∗,h∗,∗ii
//!
//! cur(f ) = cur⊥ (cur⊥ −1 (η f ) ⊗hh∗,∗i,∗i ) ∈ L(U, X ( Y ) .
//! One can prove then that the following equations hold, showing that the
//! symmetric monoidal category L is closed.
//! ev (cur(f ) ⊗ X) = f
//! cur(f ) g = cur(f (g ⊗ X))
//! cur(ev) = Id
//! where g ∈ L(V, U ).
//! It follows as usual that cur is a bijection from L(U ⊗ X, Y ) to L(U, X ( Y ).
//! We set X`Y = (X ⊥ ⊗ Y ⊥ )⊥ = X ⊥ ( Y ; this operation is the cotensor product, also called par in linear logic. Using the above properties one
//! shows that this operation is a functor L2 → L which defines another symmetric monoidal structure on L. The operation X 7→ X ⊥ is an equivalence of
//! symmetric monoidal categories from (Lop , ⊗) to (L, `).
//! 2.3.1
//!
//! MIX.
//!
//! A mix *-autonomous category is a *-autonomous category L where ⊥ is endowed
//! with a structure of commutative ⊗-monoid8 . So we have two morphisms ξ0 ∈
//! 8 If we see ⊥ as the object of scalars, which is compatible with the intuition that X ( ⊥
//! is the dual of X, that is, the “space of linear forms on X”, then this monoid structure is an
//! internal multiplication law on scalars.
//!
//! 23
//!
//!
//! --- PAGE BREAK ---
//! L(1, ⊥) and ξ2 ∈ L(⊥ ⊗ ⊥, ⊥) and some standard diagrams must commute,
//! which express that ξ0 is left and right neutral for the binary operation ξ2 ,
//! and that this binary operation is associative and commutative. Observe that
//! (ξ2 )⊥ ∈ L(⊥⊥ , (⊥ ⊗ ⊥)⊥ ) so that
//! ξ20 = (ξ2 )⊥ η1 ∈ L(1, 1`1)
//! and (1, ξ0 , ξ20 ) is a commutative `-comonoid.
//! 2.3.2
//!
//! Vectors.
//!
//! Let L be a *-autonomous category and let X1 , . . . , Xn be objects of L.
//! A (X1 , . . . , Xn )-vector is a family (uτ )τ ∈Tn where uτ ∈ L(1, `τ (X1 , . . . , Xn ))
//! satisfies uτ 0 = `ττ 0 uτ for all τ, τ 0 ∈ Tn . Of course such a vector u is determined
//! as soon as one of the uτ ’s is given. The point of this definition is that none of
//! these uτ ’s is more canonical than the others, that is why we find more convenient
//! ~ 1 , . . . , Xn ) be the set of these vectors.
//! to deal with the whole family u. Let L(X
//! Notice that, since Tn is infinite for all n, all vectors are infinite families.
//! 2.3.3
//!
//! MLL vector constructions.
//! hhi,∗i
//!
//! ~ ⊥ , X) by setting axτ = `τh∗,∗i cur(ηX −1 ⊗∗
//! Let X ∈ L. We define ax ∈ L(X
//! )∈
//! ⊥⊥
//! ⊥
//! L(1, X
//! ( X) = L(1, X `X) for all τ ∈ T2 .
//! ~ 1 , . . . , Xn , X, Y ). We define `(u) ∈ L(X
//! ~ 1 , . . . , Xn , X`Y ) as
//! Let u ∈ L(X
//! follows. Let τ ∈ Tn , we know that
//! uhτ,h∗,∗ii ∈ L(1, (`τ (X1 , . . . , Xn )`(X`Y )) = L(1, (`hτ,∗i (X1 , . . . , Xn , X`Y )) .
//! For any θ ∈ Tn+1 , we set
//! hτ,∗i
//!
//! `(u)θ = `θ
//!
//! uhτ,h∗,∗ii ∈ L(1, `θ (X1 , . . . , Xn , X`Y )) .
//!
//! One sees easily that this definition does not depend on the choice of τ : let
//! τ 0 ∈ Tn , we have
//! hτ,∗i
//!
//! `θ
//!
//! hτ,∗i
//!
//! `(u)hτ,h∗,∗ii = `θ
//!
//! hτ 0 ,∗i
//!
//! = `θ
//!
//! hτ 0 ,∗i
//!
//! `hτ,∗i `(u)hτ 0 ,h∗,∗ii
//! `(u)hτ 0 ,h∗,∗ii .
//!
//! thanks to the definition of vectors and to Equation (3).
//! Let Ui , Xi be objects of L for i = 1, 2. Given
//! ui ∈ L(1, Ui `Xi ) = L(1, Ui⊥ ( Xi )
//! for i = 1, 2, we define
//! ⊗U1 ,X1 ,U2 ,X2 (u1 , u2 ) ∈ L(1, (U1 `U2 )`(X1 ⊗ X2 ))
//!
//! 24
//!
//!
//! --- PAGE BREAK ---
//! as follows. We have cur−1 (ui ) ⊗∗hhi,∗i ∈ L(Ui⊥ , Xi ) and hence
//! v = (cur−1 (u1 ) ⊗∗hhi,∗i ) ⊗ (cur−1 (u2 ) ⊗∗hhi,∗i ) ∈ L(U1⊥ ⊗ U2⊥ , X1 ⊗ X2 ) .
//! We have
//!
//! hhi,∗i
//!
//! cur(v ⊗∗
//!
//! ) ∈ L(1, ((U1⊥ ⊗ U2⊥ ) ⊗ (X1 ⊗ X2 )⊥ )⊥ )
//!
//! So we set
//! hhi,∗i
//!
//! ⊗U1 ,X1 ,U2 ,X2 (u1 , u2 ) = (η(U1⊥ ⊗U2⊥ ) −1 ⊗ (X1 ⊗ X2 )⊥ )⊥ cur(v ⊗∗
//!
//! )
//!
//! ∈ L(1, (U1 `U2 )`(X1 ⊗ X2 ))
//! where the natural iso η is defined in Section 2.3.
//! This construction is natural in the sense that, given fi ∈ L(Xi , Xi0 ), gi ∈
//! L(Ui , Ui0 ), one has
//! ((g1 `g2 )`(f1 ⊗ f2 )) ⊗U1 ,X1 ,U2 ,X2 (u1 , u2 )
//! = ⊗U10 ,X10 ,U20 ,X20 ((f1 `g1 ) u1 , (f2 `g2 ) u2 )
//!
//! (5)
//!
//! ~ 1 , . . . , Xn , X) and v ∈ L(Y
//! ~ 1 , . . . , Yp , Y ). Let σ ∈ Tn and τ ∈ Tp .
//! Let u ∈ L(X
//! Then we have
//! uhσ,∗i ∈ L(1, (`σ (X1 , . . . , Xn ))`X)
//!
//! and
//!
//! τ
//!
//! vhτ,∗i ∈ L(1, (` (Y1 , . . . , Yp ))`Y )
//! and we set
//! ⊗(u, v)hhσ,τ i,∗i = ⊗`σ (X1 ,...,Xn ),X,`τ (Y1 ,...,Yp ),Y (uhσ,∗i , vhτ,∗i )
//! ∈ L(1, `hhσ,τ i,∗i (X1 , . . . , Xn , Y1 , . . . , Yp , X ⊗ Y )
//! since
//! (`σ (X1 , . . . , Xn )`(`τ (Y1 , . . . , Yp )))`(X ⊗ Y )
//! = `hhσ,τ i,∗i (X1 , . . . , Xn , Y1 , . . . , Yp , X ⊗ Y ) .
//! Then, given θ ∈ Tn+p+1 , one sets of course
//! hhσ,τ i,∗i
//!
//! ⊗(u, v)θ = `θ
//!
//! ⊗(u, v)hhσ,τ i,∗i .
//!
//! One checks easily that this definition does not depend on the choice of σ and τ ,
//! using Equations (3) and (5), and one can check that
//! ~ 1 , . . . , Xn , Y1 , . . . , Yp , X ⊗ Y ) .
//! ⊗(u, v) ∈ L(X
//! ~ 1 , . . . , Xn ) and ϕ ∈ Sn . Given σ ∈ Tn , we have uσ ∈
//! Let u ∈ L(X
//! σ
//! b ϕ,σ uσ ∈ L(1, `σ (Xϕ(1) , . . . , Xϕ(n) )). Given
//! L(1, ` (X1 , . . . , Xn )) and hence `
//! θ ∈ Tn , we set therefore
//! b ϕ,σ uσ ∈ L(1, `θ (Xϕ(1) , . . . , Xϕ(n) ))
//! sym(ϕ, u)θ = `σθ `
//! 25
//!
//!
//! --- PAGE BREAK ---
//! ~ ϕ(1) , . . . , Xϕ(n) ) which does not depend
//! defining an element sym(ϕ, u) of L(X
//! on the choice of σ. Indeed, let τ ∈ Tn , we know that uσ = `τσ uτ and hence
//! b ϕ,σ `τ uτ = `σ `τ `
//! b ϕ,τ uτ = `τ `
//! b ϕ,τ uτ , using Diagram (4).
//! sym(ϕ, u)θ = `σθ `
//! σ
//! θ
//! σ
//! θ
//! ~ 1 , . . . , Xn , X ⊥ ) and v ∈ L(Y
//! ~ 1 , . . . , Yp , X). We have
//! Let u ∈ L(X
//! ~ 1 , . . . , Xn , Y1 , . . . , Yp , X ⊥ ⊗ X)
//! ⊗(u, v) ∈ L(X
//! Given σ ∈ Tn and τ ∈ Tp we have
//! ⊗(u, v)hhσ,τ i,∗i ∈ L(1, ((`σ (X1 , . . . , Xn ))`(`τ Y1 , . . . , Yp ))`(X ⊥ ⊗ X))
//! so that
//! (Id`ev⊥ ) ⊗(u, v)hhσ,τ i,∗i ∈ L(1, `hhσ,τ i,hii (X1 , . . . , Xn , Y1 , . . . , Yp )) .
//! Given θ ∈ Tn+p , we set
//! hhσ,τ i,hii
//!
//! cut(u, v)θ = `θ
//!
//! (Id`ev⊥ ) ⊗(u, v)hhσ,τ i,∗i
//!
//! ~ 1 , . . . , Xn , Y1 , . . . , Yp ).
//! and we define in that way an element cut(u, v) of L(X
//! Assume now that L is a mix *-autonomous category (in the sense of Paragraph 2.3.1).
//! ~ 1 , . . . , Xn ) and
//! Let X1 , . . . , Xn and Y1 , . . . , Yp be objects of L. Let u ∈ L(X
//! ~ 1 , . . . , Yp ). Let σ ∈ Tn and τ ∈ Tp . We have uσ ∈ L(1, `σ (X1 , . . . , Xn ))
//! v ∈ L(Y
//! and vτ ∈ L(1, `τ (Y1 , . . . , Yp )). Hence
//! (uσ `vτ ) ξ20 ∈ L(1, (`σ (X1 , . . . , Xn ))`(`τ (Y1 , . . . , Yp )))
//! ~ 1 , . . . , Xn , Y1 , . . . , Yp ) by setting
//! and we define therefore mix(u, v) ∈ L(X
//! hσ,τ i
//!
//! mix(u, v)θ = `θ
//!
//! (uσ `vτ ) ξ20 ∈ L(1, `θ (X1 , . . . , Xn , Y1 , . . . , Yp ))
//!
//! for each θ ∈ Tn+p . As usual, this definition does not depend on the choice of σ
//! and τ .
//! 2.3.4
//!
//! Interpreting MLL derivations.
//!
//! We start with a valuation which, with each α ∈ A, associates [α] ∈ L in such a
//! way that [α] = [α]⊥ . We extend this valuation to an interpretation of all MLL
//! types as objects of L in the obvious manner, so that we have a De Morgan iso
//! dmA ∈ L([A⊥ ], [A]⊥ ) defined inductively as follows.
//! We set first dmα = Id[α]⊥ . We have dmA ∈ L([A⊥ ], [A]⊥ ) and dmB ∈
//! L([B ⊥ ], [B]⊥ ), therefore dmA `dmB ∈ L([(A ⊗ B)⊥ ], [A]⊥ `[B]⊥ ). We have
//! [A]⊥ `[B]⊥ = ([A]⊥⊥ ⊗ [B]⊥⊥ )⊥
//! by definition of ` and remember that η[A] ∈ L([A], [A]⊥⊥ ) and so we set
//! dmA⊗B = (η[A] ⊗ η[B] )⊥ (dmA `dmB ) .
//! 26
//!
//!
//! --- PAGE BREAK ---
//! We have [(A`B)⊥ ] = [A⊥ ] ⊗ [B ⊥ ] so dmA ⊗ dmB ∈ L([(A`B)⊥ ], [A]⊥ ⊗ [A]⊥ ).
//! By definition we have [A`B]⊥ = ([A]⊥ ⊗ [B]⊥ )⊥⊥ . So we set
//! dmA`B = η[A]⊥ ⊗[B]⊥ (dmA ⊗ dmB ) .
//! Given a sequence Γ = (A1 , . . . , An ) of types, we denote as [Γ] the sequence
//! of objects ([A1 ], . . . , [An ]).
//! Given a derivation π of a logical judgment Φ ` p : Γ we define now [π] ∈
//! ~
//! L([Γ]),
//! by induction on the structure of π.
//! Assume first that Γ = (A⊥ , A), p = ( ; x, x) and that π is the axiom
//! axiom
//! Φ`p:Γ
//! We have ax[A] ∈ L(1, [A]⊥ `[A]) and dmA ∈ L([A⊥ ], [A]⊥ ) so that we can set
//! [π]σ = `σh∗,∗i (dm−1
//! A `Id[A] ) ax[A]
//! ~
//! and we have [π] ∈ L([Γ])
//! as required.
//! −c ; →
//! −
//! Assume next that Γ = (∆, A`B), that p = (→
//! s , s`t) and that π is the
//! following derivation, where λ is the derivation of the premise:
//! −c ; →
//! −
//! Φ ` (→
//! s , s, t) : ∆, A, B
//! `-rule
//! →
//! −
//! →
//! −
//! Φ ` ( c ; s , s`t) : ∆, A`B
//! ~
//! then by inductive hypothesis we have [λ] ∈ L([∆,
//! A, B]) and hence we set
//! ~
//! [π] = `([λ]) ∈ L([∆,
//! A`B]) .
//! − →
//! →
//! −
//! −c , →
//! Assume now that Γ = (∆, Λ, A ⊗ B), that p = (→
//! d ;−
//! s , t , s ⊗ t) and π is
//! the following derivation, where λ is the derivation of the left premise and ρ is
//! the derivation of the right premise:
//! →
//! − →
//! −
//! −c ; →
//! −
//! Φ ` (→
//! s , s) : ∆, A
//! Φ ` ( d ; t , t) : Λ, B
//! ⊗-rule
//! − →
//! →
//! −
//! −c , →
//! Φ ` (→
//! d ;−
//! s , t , s ⊗ t) : ∆, Λ, A ⊗ B
//! ~
//! ~
//! then by inductive hypothesis we have [λ] ∈ L([∆,
//! A]) and [ρ] ∈ L([Λ,
//! B]) and
//! hence we set
//! ~
//! [π] = ⊗([λ], [ρ]) ∈ L([∆,
//! Λ, A ⊗ B]) .
//! −c ; s
//! Assume that ϕ ∈ S , Γ = (A
//! ,...,A
//! ), p = (→
//! ,...,s
//! ) and
//! n
//!
//! ϕ(1)
//!
//! ϕ(n)
//!
//! ϕ(1)
//!
//! ϕ(n)
//!
//! that π is the following derivation, where λ is the derivation of the premise:
//! −c ; s , . . . , s ) : A , . . . , A
//! Φ ` (→
//! 1
//! n
//! 1
//! n
//! permutation rule
//! Φ`p:Γ
//! ~ 1 ], . . . , [An ]) and we set
//! By inductive hypothesis we have [λ] ∈ L([A
//! ~
//! [π] = sym(ϕ, [λ]) ∈ L([Γ])
//! .
//! −
//! →
//! −
//! −c , →
//! −
//! Assume that Γ = (∆, Λ), that p = (→
//! d , hs | ti ; →
//! s , t ) and that π is the
//! following derivation, where λ is the derivation of the left premise and ρ is the
//! derivation of the right premise:
//! 27
//!
//!
//! --- PAGE BREAK ---
//! →
//! − →
//! −
//! −c ; →
//! −
//! Φ ` (→
//! s , s) : ∆, A⊥
//! Φ ` ( d ; t , t) : Λ, A
//! −
//! →
//! −
//! −c , →
//! −
//! Φ ` (→
//! d , hs | ti ; →
//! s , t ) : ∆, Λ
//!
//! cut rule
//!
//! ~
//! ~
//! By inductive hypothesis we have [λ] ∈ L([∆,
//! A⊥ ]) and [ρ] ∈ L([Λ,
//! A]). Let n be
//! σ
//! the length of ∆. Let σ ∈ Tn . We have [λ]hσ,∗i ∈ L(1, (` ([∆]))`[A⊥ ]) and hence
//! ~
//! (Id`dmA ) [λ]hσ,∗i ∈ L(1, (`σ ([∆]))`[A]⊥ ). We define therefore l ∈ L([∆],
//! [A]⊥ )
//! by lhσ,∗i = (Id`dmA ) [λ]hσ,∗i (this definition of l does not depend on the choice
//! of σ). We set
//! ~
//! [π] = cut(l, [ρ]) ∈ L([∆,
//! Λ]) .
//! − →
//! →
//! −
//! −c , →
//! Assume last that Γ = (∆, Λ), that p = (→
//! d ;−
//! s , t ) and that π is the
//! following derivation, where λ is the derivation of the left premise and ρ is the
//! derivation of the right premise:
//! →
//! − →
//! −
//! −c ; →
//! −
//! Φ ` (→
//! s):∆
//! Φ`(d ; t ):Λ
//! mix rule
//! − →
//! →
//! −
//! −c , →
//! Φ ` (→
//! d ;−
//! s , t ) : ∆, Λ
//! ~
//! ~
//! so that by inductive hypothesis [λ] ∈ L([∆])
//! and [ρ] ∈ L([Λ]).
//! We set
//! ~
//! [π] = mix([λ], [ρ]) ∈ L([∆,
//! Λ]) .
//! The first main property of this interpretation of derivations is that they only
//! depend on the underlying nets.
//! Theorem 3 Let π and π 0 be derivations of Φ ` p : Γ. Then [π] = [π 0 ].
//! The proof is a (tedious) induction on the structure of the derivations π and π 0 .
//! We use therefore [p] to denote the value of [π] where π is an arbitrary derivation of ` p : Γ.
//! Remark : It would be much more satisfactory to be able to define [p] directly,
//! without using the intermediate and non canonical choice of a derivation π. Such
//! a definition would use directly the fact that p fulfills a correctness criterion in
//! order to build a morphism of L. It is not very clear yet how to do that in
//! general, though such definitions are available in many concrete models of LL,
//! such as coherence spaces.
//! The second essential property of this interpretation is that it is invariant
//! under reductions (subject reduction)
//! Theorem 4 Assume that Φ ` p : Γ, Φ ` p0 : Γ and that p ; p0 . Then [p] = [p0 ].
//!
//! 2.4
//!
//! Preadditive models
//!
//! Let L be a *-autonomous category. We say that L is preadditive if each hom-set
//! L(X, Y ) is equipped with a structure of k-module (we use standard additive notations: 0 for the neutral element and + for the operation), which is compatible
//!
//! 28
//!
//!
//! --- PAGE BREAK ---
//! with composition of morphisms and tensor product:
//! 
//! 
//! !
//! X
//! X
//! X
//! 
//! νj tj 
//! µi si =
//! νj µi (tj si )
//! j∈J
//!
//! i∈I
//!
//! !
//! X
//!
//! µi si
//!
//! i∈I
//!
//! (i,j)∈I×J
//!
//! 
//! 
//! X
//! ⊗
//! νj tj  =
//! j∈J
//!
//! X
//!
//! µi νj (si ⊗ tj )
//!
//! (i,j)∈I×J
//!
//! where the µi ’s and the νj ’s are elements of k. It follows that, P
//! given a finite
//! 
//! family
//! (s
//! )
//! of
//! morphisms
//! s
//! ∈
//! L(U
//! ⊗
//! X,
//! ⊥),
//! one
//! has
//! cur
//! i
//! i∈I
//! i
//! ⊥
//! i∈I µi si =
//! P
//! i∈I µi cur⊥ (si ), and that the cotensor product is bilinear
//! 
//! ! 
//! X
//! X
//! X
//! µi si ` 
//! νj tj  =
//! µi νj (si `tj ) .
//! i∈I
//!
//! j∈J
//!
//! (i,j)∈I×J
//!
//! ~ 1 , . . . , Xn ) inherits
//! Let (Xi )ni=1 be a family of objects of L. The set L(X
//! canonically a k-module structure.
//!
//! 2.5
//!
//! Exponential structure
//!
//! If C is a category, we use Ciso to denote the category whose objects are those of
//! C and whose morphisms are the isos of C (so Ciso is a groupoid).
//! Let L be a preadditive *-autonomous category. An exponential structure
//! on L is a tuple (! , w, c, w, c, d, d) where ! is a functor Liso → Liso and the
//! other ingredients are natural transformations: wX ∈ L(!X, 1) (weakening),
//! cX ∈ L(!X, !X ⊗ !X) (contraction), wX ∈ L(1, !X) (coweakening), cX ∈ L(!X ⊗
//! !X, !X) (cocontraction), dX ∈ L(!X, X) (dereliction) and dX ∈ L(X, !X) (codereliction).
//! These morphisms are assumed moreover to satisfy the following properties.
//! The structure (!X, wX , cX , wX , cX ) is required to be a commutative bialgebra. This means that (!X, wX , cX ) is a commutative comonoid, (!X, wX , cX )
//! is a commutative monoid and that the following diagrams commute (where
//! ϕ = (1, 3, 2, 4) ∈ S4 )
//! !X ⊗ !X
//!
//! cX ⊗ cX
//!
//! (!X ⊗ !X) ⊗ (!X ⊗ !X)
//!
//! cX
//! b ϕ,hh∗,∗i,h∗,∗ii
//! ⊗
//!
//! !X
//! cX
//!
//! !X ⊗ !X
//!
//! cX ⊗ cX
//!
//! (!X ⊗ !X) ⊗ (!X ⊗ !X)
//!
//! 29
//!
//! 1
//!
//! wX
//!
//! 1
//!
//! wX
//!
//! !X
//!
//! Id1
//!
//!
//! --- PAGE BREAK ---
//! Moreover, we also require the following commutations (in the dereliction/cocontraction
//! hhi,∗i
//! h∗,hii
//! and codereliction/contraction diagrams, we omit the isos ⊗∗
//! and ⊗∗
//! for
//! the sake of readability).
//! X
//!
//! X
//!
//! dX
//!
//! !X
//!
//! 0
//!
//! 1
//!
//! wX
//!
//! X
//!
//! dX
//!
//! !X ⊗ !X
//! X
//! d X ⊗ wX + wX ⊗ d X
//!
//! !X
//!
//! 0
//!
//! 1
//!
//! dX
//!
//! !X
//!
//! dX ⊗ w X + w X ⊗ dX
//!
//! !X ⊗ !X
//!
//! wX
//!
//! cX
//!
//! dX
//!
//! !X
//! cX
//!
//! and
//! X
//!
//! dX
//!
//! !X
//!
//! IdX
//! dX
//!
//! X
//! 2.5.1
//!
//! The why not modality.
//!
//! We define ?X = (!(X ⊥ ))⊥ and we extend this operation to a functor Liso → Liso
//! in the same way (using the contravariant functoriality of ( )⊥ ). We define
//! 0
//! ⊥
//! wX
//! = wX
//! ⊥ : ⊥ → ?X .
//! ⊥
//! ⊥ ⊥
//! Since cX ⊥ : !(X ⊥ ) → !(X ⊥ ) ⊗ !(X ⊥ ), we have c⊥
//! X ⊥ : (!(X ) ⊗ !(X )) → ?X.
//! ⊥
//! ⊥
//! ⊥
//! But η!(X ⊥ ) : !(X ) → (?X) , hence (η!(X ⊥ ) ⊗ η!(X ⊥ ) ) : ?X`?X → (!(X ⊥ ) ⊗
//! !(X ⊥ ))⊥ and we set
//! ⊥
//! c0X = c⊥
//! X ⊥ (η!(X ⊥ ) ⊗ η!(X ⊥ ) ) ∈ L(?X`?X, ?X) .
//! 0
//! Then it can be shown that (?X, wX
//! , c0X ) is a commutative `-monoid (that is,
//! 0
//! a monoid in the monoidal category (L, `)). Of course, wX
//! and c0X are natural
//! transformations.
//! Last, we have dX ⊥ : !(X ⊥ ) → X ⊥ and hence (dX ⊥ )⊥ : X ⊥⊥ → ?X, so we
//! can define the natural morphism
//!
//! d0X = (dX ⊥ )⊥ ηX : X → ?X .
//! 2.5.2
//!
//! Interpreting DiLL0 derivations.
//!
//! We extend the interpretation of derivations presented in Section 2.3.4 to the
//! fragment DiLL0 presented in Section 1.4.
//! We first have to extend the interpretation of formulas – this is done in the
//! obvious way – and the definition of the De Morgan isomorphisms. We have
//! 30
//!
//!
//! --- PAGE BREAK ---
//! [(!A)⊥ ] = ?[A⊥ ] and [!A]⊥ = (![A])⊥ . By inductive hypothesis, we have the iso
//! ⊥
//! dmA : [A⊥ ] → [A]⊥ , hence ?dmA : [(!A)⊥ ] = [?(A⊥ )] → ?([A]⊥ ) = !([A]⊥⊥ )
//! ⊥
//! and since we have (!η[A] )⊥ : !([A]⊥⊥ ) → (![A])⊥ , we set
//! dm!A = (!η[A] )⊥ ?dmA ∈ Liso ([(!A)⊥ ], [!A]⊥ ) .
//! We have [(?A)⊥ ] = ![A⊥ ] and [?A]⊥ = (!([A]⊥ ))⊥⊥ so we set
//! dm?A = η!([A]⊥ ) !dmA ∈ Liso ([(?A)⊥ ], [?A]⊥ )
//! Let π be a derivation of Φ ` p : Γ, where Γ = (A1 , . . . , An ).
//! −c ; →
//! −
//! Assume first that Γ = (∆, ?A), p = (→
//! s , w) and that π is the following
//! derivation, denoting with λ the derivation of the premise:
//! −c ; →
//! −
//! Φ ` (→
//! s):Γ
//! weakening
//! →
//! −
//! →
//! −
//! Φ ` ( c ; s , w) : Γ, ?A
//! ~
//! By inductive hypothesis we have [λ] ∈ L([Γ]).
//! Let τ ∈ Tn , we have [λ]τ ∈
//! τ
//! 0
//! L(1, ` ([Γ])). We have w[A] ∈ L(⊥, ?[A]) and hence
//! 0
//! [λ]τ `w[A]
//! ∈ L(1`⊥, `hτ,∗i ([Γ, ?A]))
//!
//! so that we can set
//!
//! hτ,∗i
//!
//! [π]θ = `θ
//!
//! 0
//! ([λ]τ `w[A]
//! ) `∗h∗,hii
//!
//! for any θ ∈ Tn . The fact that the family [π] defined in that way does not depend
//! ~
//! on the choice of τ results from the fact that [λ] ∈ L([Γ]).
//! −c ; →
//! −
//! Assume that Γ = (∆, ?A), p = (→
//! s , c(t1 , t2 )) and that π is the following
//! derivation, denoting with λ the derivation of the premise:
//! −c ; →
//! −
//! ` (→
//! s , t1 , t2 ) : ∆, ?A, ?A
//! contraction
//! →
//! −
//! →
//! −
//! ` ( c ; s , c(t , t )) : ∆, ?A
//! 1
//!
//! 2
//!
//! ~
//! We have c0[A] ∈ L([?A]`[?A], [?A]). By inductive hypothesis [λ] ∈ L([∆,
//! ?A, ?A]).
//! Let τ ∈ Tn where n is the length of ∆. We have
//! [λ]hτ,h∗,∗ii ∈ L(1, (`τ ([∆]))`([?A]`[?A]))
//! and hence, given θ ∈ Tn+1 , we set
//! hτ,∗i
//!
//! [π]θ = `θ
//!
//! (`τ ([∆])`c0[A] ) [λ]hτ,h∗,∗ii
//!
//! ~
//! defining in that way [π] ∈ L([∆,
//! ?A]).
//! Assume that Γ = (!A), p = ( ; w) and that π is the following derivation
//! co-weakening
//!
//! ` ( ; w) : !A
//!
//! 31
//!
//!
//! --- PAGE BREAK ---
//! then, for θ ∈ T1 we set [π]θ = `∗θ ([!A]) w[A] defining in that way an element [π]
//! ~
//! of L([!A]).
//! − →
//! →
//! −
//! −c , →
//! Assume that Γ = (∆, Λ, !A), p = (→
//! d ;−
//! s , t , c(u, v)) and that π is the
//! following derivation
//! →
//! − →
//! −
//! −c ; →
//! −
//! Φ ` (→
//! s , u) : ∆, !A
//! Φ ` ( d ; t , v) : Λ, !A,
//! co-contraction
//! − →
//! →
//! −
//! −c , →
//! Φ ` (→
//! d ;−
//! s , t , c(u, v)) : ∆, Λ, !A
//! and we denote with λ and ρ the derivations of the two premises. By induc~
//! ~
//! tive hypothesis, we have [λ] ∈ L([∆],
//! [!A]) and [ρ] ∈ L([Λ],
//! [!A]). We have
//! ~
//! ⊗([λ], [ρ]) ∈ L([∆], [Λ], [!A] ⊗ [!A]).
//! Let m be the length of ∆ and n be the length of Λ. Let τ ∈ Tm+n , we have
//! ⊗([λ], [ρ])hτ,∗i ∈ L(1, (`τ ([∆], [Λ]))`(([!A] ⊗ [!A])). Hence, given θ ∈ Tm+n+1
//! we set
//! hτ,∗i
//! [π]θ = `θ
//! ((`τ ([∆], [Λ]))`cX ) ⊗([λ], [ρ])hτ,∗i .
//! ~
//! so that [π] ∈ L([∆],
//! [Λ], [!A]), and this definition does not depend on the choice
//! of τ .
//! −c ; →
//! −
//! Assume that Γ = (∆, ?A), p = (→
//! s , d(s)) and that π is the following
//! derivation
//! −c ; →
//! −
//! Φ ` (→
//! s , s) : ∆, A
//! dereliction
//! →
//! −
//! →
//! −
//! Φ ` ( c ; s , d(s)) : ∆, ?A
//! ~
//! Let λ be the derivation of the premise, so that [λ] ∈ L([∆],
//! [A]).
//! 0
//! We have d[A] : [A] → [?A]. Let n be the length of ∆, let τ ∈ Tn and let
//! θ ∈ Tn+1 . We set
//! hτ,∗i
//!
//! [π]θ = `θ
//!
//! ((`τ ([∆]))`d0[A] ) [λ]hτ,∗i
//!
//! ~
//! and we define in that way an element π of L([∆],
//! [?A]) which does not depend
//! on the choice of τ .
//! −c ; →
//! −
//! Assume that Γ = (∆, !A), p = (→
//! s , d(s)) and that π is the following
//! derivation
//! −c ; →
//! −
//! Φ ` (→
//! s , s) : ∆, A
//! co-dereliction
//! →
//! −
//! →
//! −
//! Φ ` ( c ; s , d(s)) : ∆, !A
//! ~
//! Let λ be the derivation of the premise, so that [λ] ∈ L([∆],
//! [A]). We have
//! d[A] : [A] → [!A]. Let n be the length of ∆, let τ ∈ Tn and let θ ∈ Tn+1 . We set
//! hτ,∗i
//!
//! [π]θ = `θ
//!
//! ((`τ ([∆]))`d[A] ) [λ]hτ,∗i
//!
//! ~
//! and we define in that way an element π of L([∆],
//! [!A]) which does not depend
//! on the choice of τ .
//! Pn
//! Last assume that p = i=1 µi pi , that π is the following derivation
//!
//! 32
//!
//!
//! --- PAGE BREAK ---
//! Φ ` pi : Γ ∀i ∈ {1, . . . , n}
//! Pn
//! Φ ` i=1 µi pi : Γ
//!
//! sum
//!
//! and that λi is the derivation of the i-th premise in this derivation. Then by
//! ~
//! inductive hypothesis we have [λi ] ∈ L([∆])
//! and we set of course
//! [π] =
//!
//! n
//! X
//!
//! µi [λi ] .
//!
//! i=1
//!
//! One can prove for this extended interpretation the same results as for the
//! MLL fragment.
//! Theorem 5 Let π and π 0 be derivations of Φ ` p : Γ. Then [π] = [π 0 ].
//! Again, we set [p] = [π] where π is a derivation of Φ ` p : Γ.
//! Theorem 6 Assume that Φ ` p : Γ, Φ ` p0 : Γ and that p ; p0 . Then [p] = [p0 ].
//!
//! 2.6
//!
//! Functorial exponential
//!
//! Let L be a preadditive *-autonomous category with an exponential structure.
//! A promotion operation on L is given by an extension of the functor ! to all
//! morphisms of L and by a lax symmetric monoidal comonad structure on the “!”
//! operation which satisfies additional conditions. More precisely:
//! • For each f ∈ L(X, Y ) we are given a morphism !f ∈ L(!X, !Y ) and the
//! correspondence f 7→ !f is functorial. This mapping f 7→ !f extends the
//! action of ! on isomorphisms.
//! • The morphisms dX , dX , wX , wX , cX and cX are natural with respect to
//! this extended functor.
//! • There is a natural transformation pX : !X → !!X which turns (!X, dX , pX )
//! into a comonad.
//! • There is a morphism µ0 : 1 → !1 and a natural transformation9 µ2X,Y :
//! !X ⊗ !Y → !(X ⊗ Y ) which satisfy the following commutations
//!
//! 1 ⊗ !X
//!
//! µ0 ⊗ !X
//!
//! !1 ⊗ !X
//!
//! ⊗hhi,∗i
//! ∗
//!
//! µ21,X
//!
//! !(1 ⊗ X)
//! !⊗hhi,∗i
//! ∗
//!
//! !X
//!
//! (6)
//!
//! 9 These morphisms are not required to be isos, whence the adjective “lax” for the monoidal
//! structure.
//!
//! 33
//!
//!
//! --- PAGE BREAK ---
//! !X ⊗ 1
//!
//! !X ⊗ µ0
//!
//! !X ⊗ !1
//!
//! µ2X,1
//!
//! !(X ⊗ 1)
//! !⊗h∗,hii
//! ∗
//!
//! ⊗h∗,hii
//! ∗
//!
//! !X
//!
//! hh∗,∗i,∗i
//!
//! (!X ⊗ !Y ) ⊗ !Z
//! µ2X,Y
//!
//! ⊗h∗,h∗,∗ii
//!
//! !X ⊗ (!Y ⊗ !Z)
//!
//! ⊗ !Z
//!
//! !(X ⊗ Y ) ⊗ !Z
//!
//! µ2X⊗Y,Z
//!
//! (7)
//!
//! !X ⊗ µ2Y,Z
//!
//! !X ⊗ !(Y ⊗ Z)
//! µ2X,Y ⊗Z
//!
//! hh∗,∗i,∗i
//!
//! !((X ⊗ Y ) ⊗ Z)
//!
//! σ!X,!Y
//!
//! !X ⊗ !Y
//! µ2X,Y
//!
//! !⊗h∗,h∗,∗ii
//!
//! !(X ⊗ (Y ⊗ Z))
//!
//! (8)
//!
//! !Y ⊗ !X
//! µ2Y,X
//!
//! !σX,Y
//!
//! !(X ⊗ Y )
//!
//! !(Y ⊗ X)
//!
//! (9)
//!
//! • The following diagrams commute
//!
//! !X ⊗ !Y
//!
//! µ2X,Y
//!
//! !(X ⊗ Y )
//!
//! 1
//!
//! dX⊗Y
//!
//! dX ⊗ dY
//!
//! µ0
//!
//! !1
//! d1
//!
//! 1
//!
//! X ⊗Y
//!
//! 1
//!
//! µ2X,Y
//!
//! !X ⊗ !Y
//!
//! !(X ⊗ Y )
//! pX⊗Y
//!
//! pX ⊗ pY
//!
//! !!X ⊗ !!Y
//!
//! µ2!X,!Y
//!
//! !(!X ⊗ !Y )
//!
//! !µ2X,Y
//!
//! !!(X ⊗ Y )
//!
//! 1
//!
//! µ0
//!
//! !1
//! p1
//!
//! µ0
//! !µ0
//!
//! !1
//!
//! !!1
//!
//! When these conditions hold, one says that (µ0 , µ2 ) is a lax symmetric monoidal
//! structure on the comonad (!, d, p).
//! 2.6.1
//!
//! Monoidality and structural morphisms.
//!
//! This monoidal structure must also be compatible with the structural constructions.
//!
//! 34
//!
//!
//! --- PAGE BREAK ---
//! !X ⊗ !Y
//! wX ⊗ wY
//!
//! 1⊗1
//!
//! cX ⊗ cY
//!
//! !X ⊗ !Y
//!
//! µ2X,Y
//!
//! !(X ⊗ Y )
//!
//! µ0
//!
//! 1
//!
//! !1
//! w1
//!
//! wX⊗Y
//!
//! hhi,hii
//!
//! 1
//!
//! ⊗hi
//!
//! 1
//!
//! 1
//!
//! (!X ⊗ !X) ⊗ (!Y ⊗ !Y )
//!
//! hi
//!
//! ⊗hhi,hii
//!
//! b ϕ,σ
//! ⊗
//! µ2X,Y
//!
//! 1
//!
//! (!X ⊗ !Y ) ⊗ (!X ⊗ !Y )
//!
//! µ0
//!
//! µ2X,Y ⊗ µ2X,Y
//!
//! !(X ⊗ Y )
//!
//! cX⊗Y
//!
//! 1⊗1
//! µ0 ⊗ µ0
//!
//! !1
//!
//! c1
//!
//! !1 ⊗ !1
//!
//! !(X ⊗ Y ) ⊗ !(X ⊗ Y )
//!
//! where ϕ = (1, 3, 2, 4) ∈ S4 and σ = hh∗, ∗i, h∗, ∗ii.
//! 2.6.2
//!
//! Monoidality and costructural morphisms.
//!
//! We need the following diagrams to commute in order to validate the reduction
//! rules of DiLL.
//!
//! (!X ⊗ !X) ⊗ !Y
//! 1 ⊗ !Y
//!
//! wX ⊗ !Y
//!
//! !X ⊗ !Y
//!
//! cX ⊗ !Y
//!
//! !X ⊗ !Y
//!
//! (!X ⊗ !X) ⊗ cY
//!
//! (!X ⊗ !X) ⊗ (!Y ⊗ !Y )
//!
//! 1 ⊗ wY
//! µ2X,Y
//!
//! 1⊗1
//!
//! µ2X,Y
//!
//! b ϕ,σ
//! ⊗
//!
//! hhi,hii
//!
//! (!X ⊗ !Y ) ⊗ (!X ⊗ !Y )
//!
//! ⊗hi
//!
//! 1
//!
//! wX⊗Y
//!
//! !(X ⊗ Y )
//!
//! µ2X,Y ⊗ µ2X,Y
//!
//! !(X ⊗ Y ) ⊗ !(X ⊗ Y )
//!
//! cX⊗Y
//!
//! !(X ⊗ Y )
//!
//! where ϕ = (1, 3, 2, 4) ∈ S4 and σ = hh∗, ∗i, h∗, ∗ii.
//! X ⊗ !Y
//!
//! dX ⊗ !Y
//!
//! !X ⊗ !Y
//! µ2X,Y
//!
//! X ⊗ dY
//!
//! X ⊗Y
//! 2.6.3
//!
//! dX⊗Y
//!
//! !(X ⊗ Y )
//!
//! Digging and structural morphisms.
//!
//! We assume that pX is a comonoid morphism from (!X, wX , cX ) to (!!X, w!X , c!X ),
//! in other words, the following diagrams commute.
//! 35
//!
//!
//! --- PAGE BREAK ---
//! pX
//!
//! !X
//!
//! !!X
//!
//! c!X
//! pX ⊗ pX
//!
//! !X ⊗ !X
//!
//! 1
//! 2.6.4
//!
//! !!X
//!
//! cX
//!
//! w!X
//!
//! wX
//!
//! pX
//!
//! !X
//!
//! !!X ⊗ !!X
//!
//! Digging and costructural morphisms.
//!
//! It is not required that pX be a monoid morphism from (!X, wX , cX ) to (!!X, w!X , c!X ),
//! but the following diagrams must commute.
//!
//! 1
//!
//! wX
//!
//! !X ⊗ !X
//!
//! !X
//! pX
//!
//! µ0
//!
//! !1
//!
//! !wX
//!
//! cX
//!
//! !X
//!
//! pX
//!
//! !!X
//!
//! pX ⊗ pX
//!
//! !cX
//! µ2!X,!X
//!
//! !!X ⊗ !!X
//!
//! !!X
//!
//! !(!X ⊗ !X)
//!
//! In the same spirit, we need a last diagram to commute, which describes the
//! interaction between codereliction and digging.
//! dX
//!
//! X
//!
//! pX
//!
//! !X
//!
//! !!X
//!
//! ⊗∗
//! hhi,∗i
//!
//! !cX
//!
//! 1⊗X
//! 2.6.5
//!
//! w X ⊗ dX
//!
//! !X ⊗ !X
//!
//! pX ⊗ d!X
//!
//! !!X ⊗ !!X
//!
//! Preadditive structure and functorial exponential.
//!
//! Our last requirement justifies the term “exponential” since it expresses that
//! sums are turned into products by this functorial operation.
//! !X
//!
//! !0
//!
//! !(f + g)
//!
//! !X
//!
//! !X
//!
//! !Y
//!
//! cX
//!
//! wX
//!
//! wX
//!
//! 1
//!
//! !X ⊗ !X
//!
//! cX
//! !f ⊗ !g
//!
//! !Y ⊗ !Y
//!
//! Remark : There is another option in the categorical axiomatization of models
//! of Linear Logic that we briefly describe as follows.
//! • One requires the linear category L to be cartesian, with a terminal object
//! > and a cartesian product usually denoted as X1 & X2 , projections πi ∈
//! L(X1 & X2 , Xi ) and pairing hf1 , f2 i ∈ L(Y, X1 & X2 ) for fi ∈ L(Y, Xi ).
//! This provides in particular L with another symmetric monoidal structure.
//! • As above, one require the functor ! to be a comonad. But we equip it
//! now with a symmetric monoidal structure (m0 , m2 ) from the monoidal
//! 36
//!
//!
//! --- PAGE BREAK ---
//! category (L, &) to the monoidal category (L, ⊗). This means in particular that m0 ∈ L(1, !>) and m2X1 ,X2 ∈ L(!X1 ⊗ !X2 , !(X1 & X2 )) are isos.
//! These isos are often called Seely isos in the literature, though Girard already stressed their significance in [Gir87], admittedly not in the general
//! categorical setting of monoidal comonads. An additional commutation is
//! required, which describes the interaction between m2 and p.
//! Using this structure, the comonad (! , d, p) can be equipped with a lax symmetric
//! monoidal structure (µ0 , µ2 ). Again, our main reference for these notions and
//! constructions is [Mel09]. In this setting, the structural natural transformations
//! wX and cX can be defined and it is well known that the Kleisli category L! of
//! the comonad ! is cartesian closed.
//! If we require the category L to be preadditive in the sense of Section 2.4,
//! it is easy to see that > is also an initial object and that & is also a coproduct.
//! Using this fact, the natural transformations wX and cX can also be defined.
//! To describe a model of DiLL in this setting, one has to require these Seely
//! monoidality isomorphisms to satisfy some commutations with the d natural
//! transformation.
//! Here, we prefer a description which does not use cartesian products because
//! it is closer to the basic constructions of the syntax of proof-structures and makes
//! the presentation of the semantics conceptually simpler and more canonical, to
//! our taste at least.
//! 2.6.6
//!
//! Generalized monoidality, contraction and digging.
//!
//! Just as the monoidal structure of a monoidal category, the monoidal structure
//! of ! can be parameterized by monoidal trees. Let n ∈ N and let τ ∈ Tn . Given
//! →
//! −
//! −
//! −
//! τ
//! τ →
//! τ →
//! a family of objects X = (X1 , . . . , Xn ) of L, we define µ→
//! − : ⊗ (! X ) → !⊗ ( X )
//! X
//! by induction on τ as follows:
//! µhi = µ0
//! µ∗X = Id!X
//! hσ,τ i
//!
//! 2
//! σ
//! τ
//! µ→
//! −
//! →
//! − (µ→
//! − ⊗ µ→
//! −).
//! − →
//! − =µ σ →
//! ⊗ ( X ),⊗τ ( Y )
//! X
//! Y
//! X,Y
//!
//! Given σ, τ ∈ Tn and ϕ ∈ Sn , one can prove that the following diagrams
//! commute
//!
//! →
//! −
//! ⊗σ (! X )
//! σ
//! µ→
//! −
//! X
//!
//! →
//! −
//! !⊗σ ( X )
//!
//! →
//! −
//! ⊗σ
//! τ (! X )
//!
//! →
//! −
//! !⊗σ
//! τ (! X )
//!
//! →
//! −
//! ⊗τ (! X )
//!
//! →
//! −
//! ⊗σ (! X )
//!
//! τ
//! µ→
//! −
//!
//! −
//! b ϕ,σ (!→
//! ⊗
//! X)
//!
//! µσ →
//! −
//!
//! σ
//! µ→
//! −
//!
//! X
//!
//! X
//!
//! →
//! −
//! !⊗τ ( X )
//!
//! →
//! −
//! !⊗σ ( X )
//!
//! 37
//!
//! →
//! −
//! ⊗σ (ϕ(!
//! b X ))
//!
//! ϕ,σ
//!
//! b
//! !⊗
//!
//! →
//! −
//! (X )
//!
//! ϕ(
//! b X)
//!
//! →
//! −
//! !⊗σ (ϕ(
//! b X ))
//!
//!
//! --- PAGE BREAK ---
//! σ
//!
//! −)
//! X
//! →
//! − ⊗ (w→
//! →
//! −
//! ⊗σ (! X )
//! ⊗σ 1 = ⊗σ0
//!
//! →
//! −
//! ⊗σ (! X )
//!
//! σ
//!
//! σ
//! µ→
//! −
//! X
//!
//! →
//! −
//! !⊗σ ( X )
//!
//! −
//! w⊗σ (→
//! X)
//!
//! −)
//! ⊗σ (d→
//! X
//!
//! →
//! −
//! ⊗σ X
//!
//! σ
//! µ→
//! −
//!
//! ⊗hi0
//!
//! X
//!
//! →
//! −
//! !⊗σ ( X )
//!
//! 1 = ⊗hi
//!
//! −
//! d⊗σ (→
//! X)
//!
//! →
//! −
//! where 1 is the sequence (1, . . . , 1) (n elements) and σ0 = σ [hi/∗] ∈ T0 (the tree
//! obtained from σ by replacing each occurrence of ∗ by hi).
//! Before stating the next commutation, we define a generalized form of contrac−
//! − →
//! −
//! σ
//! σ →
//! hσ,σi →
//! tion c→
//! (! X , ! X ) as the following composition of morphisms:
//! − : ⊗ !X → ⊗
//! X
//!
//! −
//! σ →
//!
//! ⊗ !X
//!
//! −
//! ⊗σ c→
//! X
//!
//! −
//! →
//! ⊗ !X 0
//! σ2
//!
//! b ϕ,σ
//! ⊗
//!
//! σ
//!
//! 2
//! ⊗hσ,σi
//!
//! →
//! − →
//! −
//! ⊗ (! X , ! X )
//! σ2
//!
//! →
//! − →
//! −
//! ⊗hσ,σi (! X , ! X )
//!
//! −
//! →
//! where X 0 = (X1 , X1 , X2 , X2 , . . . , Xn , Xn ), σ2 = σ [h∗, ∗i/∗] and ϕ ∈ S2n is
//! defined by ϕ(2i + 1) = i + 1 and ϕ(2i + 2) = i + n + 1 for i ∈ {0, . . . , n − 1}.
//! With these notations, one can prove that
//! →
//! −
//! ⊗σ ! X
//!
//! σ
//! c→
//! −
//!
//! →
//! − →
//! −
//! →
//! −
//! →
//! −
//! ⊗hσ,σi (! X , ! X ) = (⊗σ ! X ) ⊗ (⊗σ ! X )
//!
//! X
//!
//! σ
//! µ→
//! −
//! X
//!
//! →
//! −
//! !⊗σ X
//!
//! σ
//! σ
//! µ→
//! − ⊗ µ→
//! −
//! −
//! c⊗σ (→
//! X)
//!
//! X
//!
//! X
//!
//! →
//! −
//! →
//! −
//! (!⊗σ X ) ⊗ (!⊗σ X )
//!
//! −
//! −
//! σ
//! σ →
//! σ →
//! We also define a generalized version of digging p→
//! − : ⊗ ! X → !⊗ ! X as the
//! X
//! following composition of morphisms:
//! σ
//!
//! −
//! X
//! →
//! − ⊗ p→
//! →
//! −
//! ⊗σ ! X
//! ⊗σ !! X
//!
//! µσ→
//! −
//!
//! →
//! −
//! !⊗σ ! X
//!
//! !X
//!
//! With this notation, one can prove that
//! →
//! −
//! ⊗σ ! X
//!
//! σ
//! p→
//! −
//!
//! →
//! −
//! !⊗σ ! X
//!
//! X
//!
//! σ
//! µ→
//! −
//!
//! σ
//! !µ→
//! −
//!
//! X
//!
//! −
//! σ→
//!
//! −
//! p⊗σ →
//! X
//!
//! !⊗ X
//!
//! X
//!
//! −
//! σ→
//!
//! !!⊗ X
//!
//! We have phi = µ0 , p∗X = pX , and observe that the following generalizations of
//! the comonad laws hold. The two commutations involving digging and dereliction
//! generalize to:
//! →
//! −
//! ⊗σ ! X
//! σ
//! →
//! −
//! ⊗σ ! X
//!
//! µ→
//! −
//!
//! σ
//! p→
//! −
//!
//! X
//!
//! X
//!
//! →
//! −
//! ⊗σ ! X
//!
//! −
//! d⊗σ !→
//! X
//!
//! →
//! −
//! !⊗σ ! X
//!
//! 38
//!
//! −
//! !⊗σ d→
//! X
//!
//! →
//! −
//! !⊗σ X
//!
//!
//! --- PAGE BREAK ---
//! →
//! −
//! The square diagram involving digging generalizes as follows. Let Y = (Y1 , . . . , Ym )
//! be another list of objects and let τ ∈ Tm . One can prove that
//! σ
//! p→
//! −
//!
//! −
//! σ →
//!
//! →
//! −
//! !⊗σ ! X
//!
//! X
//!
//! ⊗ !X
//! σ
//! p→
//! −
//!
//! σ
//! !p→
//! −
//!
//! X
//!
//! −
//! σ →
//!
//! X
//!
//! −
//! p⊗σ !→
//! X
//!
//! !⊗ ! X
//!
//! −
//! σ →
//!
//! !!⊗ ! X
//!
//! and then one can generalize this property as follows
//! hσ,τ i
//!
//! −
//! σ →
//!
//! p→
//! − →
//! −
//!
//! −
//! τ →
//!
//! →
//! −
//! →
//! −
//! !((⊗σ ! X ) ⊗ (⊗τ ! Y ))
//!
//! X ,Y
//!
//! (⊗ ! X ) ⊗ (⊗ ! Y )
//! −
//! σ
//! τ →
//! p→
//! − ⊗ (⊗ ! Y )
//! X
//!
//! −
//! σ →
//!
//! −
//! τ →
//!
//! p
//!
//! −
//! σ
//! τ →
//! !(p→
//! − ⊗ (⊗ ! Y ))
//!
//! h∗,τ i
//! →
//! − →
//! −
//! ⊗σ ! X , Y
//!
//! X
//!
//! −
//! σ →
//!
//! →
//! −
//! !(!(⊗ ! X ) ⊗ (⊗τ ! Y ))
//!
//! !(⊗ ! X ) ⊗ (⊗ ! Y )
//!
//! (10)
//!
//! 2.6.7
//!
//! Generalized promotion and structural constructions.
//! →
//! −
//! →
//! −
//! Let f : ⊗σ ! X → Y , we define the generalized promotion f ! : ⊗σ ! X → !Y by
//! σ
//! f ! = !f p→
//! − . Using the commutations of Section 2.6.6, one can prove that this
//! X
//! construction obeys the following commutations.
//! f!
//!
//! −
//! σ →
//!
//! !Y
//!
//! ⊗ !X
//! σ
//!
//! −
//! ⊗ w→
//! X
//!
//! wY
//!
//! σ
//!
//! ⊗hi0
//!
//! →
//! −
//! ⊗σ 1 = ⊗σ0
//!
//! ⊗hi = 1
//!
//! with the same notations as before.
//! With these notations, we have
//! f!
//!
//! −
//! σ →
//!
//! !Y
//!
//! ⊗ !X
//! σ
//! c→
//! −
//! X
//!
//! cY
//! f! ⊗ f!
//!
//! →
//! − →
//! −
//! ⊗hσ,σi (! X , ! X )
//!
//! !Y ⊗ !Y
//!
//! The next two diagrams deal with the interaction between generalized promotion and dereliction (resp. digging).
//! f!
//!
//! →
//! −
//! ⊗σ ! X
//!
//! !Y
//! dY
//! f
//!
//! Y
//!
//! 39
//!
//!
//! --- PAGE BREAK ---
//! −
//! σ →
//!
//! −
//! τ →
//!
//! →
//! −
//! f ! ⊗ (⊗τ ! Y )
//!
//! (⊗ ! X ) ⊗ (⊗ ! Y )
//!
//! →
//! −
//! !Y ⊗ (⊗τ ! Y )
//!
//! hσ,τ i
//!
//! p→
//! − →
//! −
//! X ,Y
//!
//! →
//! −
//! →
//! −
//! !((⊗σ ! X ) ⊗ (⊗τ ! Y ))
//!
//! !
//!
//! −
//! τ →
//!
//! !(f ⊗ (⊗ ! Y ))
//!
//! p
//!
//! h∗,τ i
//! →
//! −
//! Y, Y
//!
//! →
//! −
//! !(!Y ⊗ (⊗τ ! Y ))
//!
//! The second diagram follows easily from (10) and allows one to prove the fol−
//! →
//! −
//! →
//! lowing property. Let f : ⊗σ !X → Y and g : !Y ⊗ (⊗τ !Y ) → Z so that
//! −
//! →
//! −
//! →
//! f ! : ⊗σ !X → !Y and g ! : !Y ⊗ (⊗τ !Y ) → !Z; one has
//! −
//! →
//! −
//! →
//! →
//! −
//! →
//! −
//! g ! (f ! ⊗ (⊗τ ! Y )) = (g (f ! ⊗ (⊗τ ! Y )))! : (⊗σ !X) ⊗ (⊗τ !Y ) → !Z
//! Remark : We actually need a more general version of this property, where f ! is
//! not necessarily in leftmost position in the ⊗ tree. It is also easy to obtain, but
//! notations are more heavy. We use the same kind of convention in the sequel but
//! remember that the corresponding properties are easy to generalize.
//! 2.6.8
//!
//! Generalized promotion and costructural constructions.
//! →
//! −
//! →
//! −
//! →
//! −
//! Let f : !X⊗(⊗σ ! X ) → Y . Observe that f (wX ⊗(⊗σ ! X )) ⊗σhhi,σi −→ : ⊗σ ! X → Y .
//! !X
//! The following equation holds:
//! →
//! −
//! −
//! σ →
//! σ
//! !
//! f ! (wX ⊗ (⊗σ ! X )) ⊗σhhi,σi −
//! → = (f (wX ⊗ (⊗ ! X )) ⊗hhi,σi −
//! →)
//! !X
//! !X
//! →
//! −
//! →
//! −
//! Similarly, we have f (cX ⊗(⊗σ ! X )) : (!X ⊗ !X)⊗(⊗σ ! X ) → Y and the following
//! equation holds:
//! →
//! −
//! →
//! −
//! f ! (cX ⊗ (⊗σ ! X )) = (f (cX ⊗ (⊗σ ! X )))!
//! This results from the commutations of Sections 2.6.2 and 2.6.4.
//! 2.6.9
//!
//! Generalized promotion and codereliction (also known as chain
//! rule).
//! →
//! −
//! Let f : !X ⊗ (⊗σ ! X ) → Y . We set
//! →
//! −
//! →
//! −
//! f0 = f (wX ⊗ (⊗σ ! X )) ⊗σhhi,σi : ⊗σ ! X → Y
//! Then we have
//! −
//! σ →
//!
//! X ⊗ (⊗ ! X )
//!
//! →
//! −
//! dX ⊗ (⊗σ ! X )
//!
//! →
//! −
//! !X ⊗ (⊗σ ! X )
//!
//! f!
//!
//! !Y
//!
//! σ
//! dX ⊗ c→
//! −
//! X
//!
//! −
//! σ →
//!
//! →
//! −
//! !X ⊗ ((⊗ ! X ) ⊗ (⊗σ ! X ))
//! h∗,h∗,∗ii
//! ⊗hh∗,∗i,∗i
//!
//! →
//! −
//! →
//! −
//! (!X ⊗ (⊗σ ! X )) ⊗ (⊗σ ! X )
//!
//! cY
//!
//! f ⊗ f0!
//!
//! Y ⊗ !Y
//!
//! dY ⊗ !Y
//!
//! !Y ⊗ !Y
//!
//! This results from the commutations of Sections 2.6.2 and 2.6.4.
//! 40
//!
//!
//! --- PAGE BREAK ---
//! 2.6.10
//!
//! Interpreting DiLL derivations.
//!
//! For the sake of readability, we assume here that the De Morgan isomorphisms
//! (see 2.3.4) are identities, so that [A⊥ ] = [A]⊥ for each formula A. The general
//! definition of the semantics can be obtained by inserting De Morgan isomorphisms at the correct places in the forthcoming expressions.
//! →
//! −
//! −
//! Let P be a net of arity n+1 and let pi = (→
//! ci ; ti , ti ) for i = 1, . . . , n. Consider
//! the following derivation π, where we denote as λ, ρ1 , . . . , ρn the derivations of
//! the premises.
//! ⊥
//! Φ ` P : ?A⊥
//! Φ ` p1 : Γ1 , !A1 · · · Φn ` pn : Γn , !An
//! 1 , . . . , ?An , B
//! →
//! −
//! →
//! − !(n)
//! →
//! −
//! →
//! −
//! Φ ` ( c1 , . . . , cn ; t1 , . . . , tn , P
//! (t1 , . . . , tn )) : Γ1 , . . . , Γn , !B
//! ⊥
//! ⊥
//! ~
//! By inductive hypothesis, we have [λ] ∈ L((![A
//! 1 ]) , . . . , (![An ]) , [B]) so that,
//! picking an element σ of Tn we have
//!
//! [λ]hσ,∗i ∈ L(1, `σ ((![A1 ])⊥ , . . . , (![An ])⊥ )`[B])
//! = L(1, ⊗σ (![A1 ], . . . , ![An ]) ( [B])
//! and hence
//! (cur−1 ([λ]hσ,∗i ) ⊗σhhi,σi )! ∈ L(⊗σ (![A1 ], . . . , ![An ]), ![B]) .
//! ~ i ], ![Ai ]). Let li be the length of Γi ,
//! For i = 1, . . . , n, we have [ρi ] ∈ L([Γ
//! and let us choose τi ∈ Tli . We have [ρi ]hτi ,∗i ∈ L(1, `τi ([Γi ])`![Ai ]) and hence,
//! setting
//! i
//! ri = cur−1 ([ρi ]hτi ,∗i ) ⊗τhhi,τ
//! ∈ L(⊗τi ([Γi ])⊥ , ![Ai ])
//! ii
//! −
//! we have ⊗σ (→
//! r ) ∈ L(⊗θ ([∆]⊥ ), ⊗σ (![A ], . . . , ![A ])) where
//! 1
//!
//! n
//!
//! ∆ = Γ1 , . . . , Γn
//! θ = σ(τ1 , . . . , τn )
//! where σ(τ1 , . . . , τn ) (for σ ∈ Tn and τi ∈ Tni for i = 1, . . . , k) is the element of
//! Tn1 +···+nk defined inductively by
//! hi() = hi
//! ∗(τ ) = τ
//! 0
//!
//! hσ, σ i(τ1 , . . . , τn ) = hσ(τ1 , . . . , τk ), σ 0 (τk+1 , . . . , τn )i
//! where σ ∈ Tk , σ 0 ∈ Tn−k
//! We have therefore
//! −
//! (cur−1 ([λ]hσ,∗i ) ⊗σhhi,σi )! ⊗σ (→
//! r ) ∈ L(⊗θ ([∆]⊥ ), [B])
//! We set
//! hhi,θi
//! −
//! [π]θ = cur((cur−1 ([λ]hσ,∗i ) ⊗σhhi,σi )! ⊗σ (→
//! r ) ⊗θ
//! ) ∈ L(1, `hθ,∗i ([∆, !B]))
//!
//! ~
//! and this gives us a definition of [π] ∈ L([∆,
//! !B]) which does not depend on the
//! choice of θ.
//! 41
//!
//!
//! --- PAGE BREAK ---
//! Theorem 7 Let π and π 0 be derivations of Φ ` p : Γ. Then [π] = [π 0 ].
//! Again, we set [p] = [π] where π is a derivation of Φ ` p : Γ.
//! Theorem 8 Assume that Φ ` p : Γ, Φ ` p0 : Γ and that p ; p0 . Then [p] = [p0 ].
//! The proofs of these results are tedious inductions, using the commutations
//! described in paragraphs 2.6.7, 2.6.8 and 2.6.9.
//!
//! 2.7
//!
//! The differential λ-calculus
//!
//! Various λ-calculi have been proposed, as possible extensions of the ordinary λcalculus with constructions corresponding to the above differential and costructural rules of differential LL. We record here briefly our original syntax of [ER03],
//! simplified by Vaux in [Vau05]10 .
//! A simple term is either
//! • a variable x,
//! • or an ordinary application (M ) R where M is a simple terms and R is a
//! term,
//! • or an abstraction λx M where x is a variable and M is a simple term,
//! • or a differential application DM · N where M and N are simple terms.
//! A term is a finite linear combination of simple terms, with coefficients in k.
//! Substitution of a term R for a variable x in a simple term M , denoted as
//! M [R/x] is defined as usual, whereas differential (or linear) substitution of a
//! simple term for a variable in another simple term, denoted as ∂M
//! ∂x · N , is defined
//! as follows:
//! (
//! t if x = y
//! ∂y
//! ·t=
//! ∂x
//! 0 otherwise
//! ∂λy M
//! ∂M
//! · N = λy
//! ·N
//! ∂x
//!  ∂x
//! 
//! 
//! 
//! ∂M
//! ∂N
//! ∂DM · N
//! ·P =D
//! · P · N + DM ·
//! ·P
//! ∂x
//! ∂x
//! ∂x
//! 
//! 
//! 
//! 
//! 
//! ∂ (M ) R
//! ∂M
//! ∂R
//! ·N =
//! · N R + DM ·
//! ·N
//! R
//! ∂x
//! ∂x
//! ∂x
//! All constructions are linear, except for ordinary application which is not linear
//! in the argument. This means that when we write e.g. (M1 + M2 ) R, what we
//! actually intend is (M1 ) R + (M2 ) R. Similarly, substitution M [R/x] is linear in
//! 10 Alternative syntaxes have been proposed, which are formally closer to Boudol’s calculus
//! with multiplicities or with resources and are therefore often called resource λ-calculi
//!
//! 42
//!
//!
//! --- PAGE BREAK ---
//! M and not in R, whereas differential substitution ∂M
//! ∂x · N is linear in both M
//! and N . There are two reduction rules:
//! (λx M ) R β M [R/x]
//! 
//! 
//! ∂M
//! ·N
//! D(λx M ) · N βd λx
//! ∂x
//! which have of course to be closed under arbitrary contexts. The resulting calculus can be proved to be Church-Rosser using fairly standard techniques (Tait
//! - Martin-Löf), to have good normalization properties in the typed case etc,
//! see [ER03, Vau05]. To be more precise, Church-Rosser holds only up to the least
//! congruence on terms which identifies D(DM · N1 ) · N2 and D(DM · N2 ) · N1 , a
//! syntactic version of Schwarz Lemma: terms are always considered up to this
//! congruence called below symmetry of derivatives.
//! 2.7.1
//!
//! Resource calculus.
//!
//! Differential application can be iterated: given simple terms M, N1 , . . . , Nn , we
//! define Dn M · (N1 , . . . , Nn ) = D(· · · DM · N1 · · · ) · Nn ; the order on the terms
//! N1 , . . . , Nn does not matter, by symmetry of derivatives. The (general) resource
//! calculus is another syntax for the differential λ-calculus, in which the combination (Dn M · (N1 , . . . , Nn )) R is considered as one single operation denoted e.g. as
//! M [N1 , . . . , Nn , R∞ ] where the superscript ∞ is here to remind that R can be
//! arbitrarily duplicated during reduction, unlike the Ni ’s. This presentation of
//! the calculus, studied in particular by Tranquilli and Pagani, and also used for
//! instance in [BCEM11], has very good properties as well. It is formally close
//! to Boudol’s λ-calculus with multiplicities such as presented in [BCL99], with
//! the difference that the operational semantics of Boudol’s calculus is given as a
//! rewriting strategy whereas in the differential version of the resource λ-calculus,
//! redexes can be reduced everywhere in terms. The price to pay is that reduction becomes non-deterministic in the sense that it can produce formal sums of
//! terms.
//! 2.7.2
//!
//! The finite resource calculus.
//!
//! If, in the resource calculus above, one restricts one’s attention to the terms
//! where all applications are of the form
//! M [N1 , . . . , Nn , 0∞ ]
//! which corresponds to the differential term (D(· · · DM · N1 · · · ) · Nn ) 0, then one
//! gets a calculus which is stable under reduction and where all terms are strongly
//! normalizing. This calculus, called the finite resource calculus, can be presented
//! as follows.
//! • Any variable x is a term.
//! • If x is a variable and s is a simple term then λx s is a simple term.
//! 43
//!
//!
//! --- PAGE BREAK ---
//! • If S is a finite multiset (also called bunch in the sequel) of simple terms
//! then hsi S is a simple term. Intuitively, this term stands for the application
//! s[s1 , . . . , sn , 0∞ ] of the resource calculus, where [s1 , . . . , sn ] = S.
//! A term is a (possibly infinite11 ) linear combination of finite terms. This syntax
//! is extended from simple terms to general terms by linearity. For instance, the
//! term hxi [y + z, y + z] stands for hxi [y, y] + 2 hxi [y, z] + hxi [z, z].
//! In the finite resource calculus, it is natural to perform several βd -reductions
//! in one step, and one gets
//! (P
//! 
//! 
//! if degs x = n
//! f ∈Sn s sf (1) /x1 , . . . , sf (n) /xn
//! hλx si S βd
//! 0
//! otherwise
//! where d = degx s is the number of occurrences of x in s (which is a simple term),
//! x1 , . . . , xd are the occurrences of x in s and the multiset S is [s1 , . . . , sn ].
//! Again, this calculus enjoys confluence, and also strong normalization (even
//! in the untyped case). It can be used for hereditarily Taylor expanding λ-terms
//! as explained in [ER08, ER06, Ehr10].
//! Taylor expansion consists in hereditarily replacing, in a differential λ-term,
//! any ordinary application (M ) N by the infinite sum
//! ∞
//! X
//! 1
//!
//! n!
//! n=0
//!
//! (Dn M · (N, . . . , N )) 0 .
//!
//! More precisely, it is a transformation M 7→ M ∗ from resource terms to finite
//! resource terms which is defined as follows:
//! x∗ = x
//! ∗
//!
//! (λx M ) = λx (M ∗ )
//! ∞
//! X
//! 1
//! ∞ ∗
//! hM ∗ i [N1 ∗ , . . . , Nn ∗ , R∗ , . . . , R∗ ]
//! M [N1 , . . . , Nn , R ] =
//! p!
//! p=0
//! so that the Taylor expansion of a resource term is a generally infinite linear combination of finite resource terms. In the definition above, we use the extension
//! by linearity of the syntax of finite resource terms to arbitrary (possibly infinite)
//! linear combinations. The coefficients belong to the considered semi-ring k where
//! division by positive natural numbers must be possible.
//! In [ER08, ER06] we studied the behavior of this expansion with respect to
//! differential β-reduction in the case where the expanded terms come from the
//! λ-calculus (that is, do not contain differential applications; this is the uniform
//! case), and we exhibited tight connections between this operation and Krivine’s
//! machine, an implementation of linear head reduction.
//! 11 When considering infinite linear combinations, one has to deal with the possibility of
//! unbounded coefficients appearing during the reduction. One option is to accept infinite coefficients, but it is also possible to prevent this phenomenon to occur by topological means as
//! explained in [Ehr10].
//!
//! 44
//!
//!
//! --- PAGE BREAK ---
//! There is a simple translation from resource terms (or differential terms) to
//! DiLL proof-nets. When restricted to the finite resource calculus, this translation
//! ranges in DiLL0 . This translation extends Girard’s Translation from the λcalculus to LL proof-nets.
//!
//! 3
//!
//! More on exponential structures
//!
//! We address here two aspects of exponential structures: we study a simple condition expressing that a map whose derivative is uniformly equal to 0 must be
//! constant, and we propose an axiomatization of antiderivatives in this categorical
//! setting.
//! So we assume to be given a preadditive *-autonomous category L equipped
//! with an exponential structure in the sense of Section 2.5, and we use the same
//! notations as in this section.
//! For the sake of notational simplicity, we do as if ⊗ were strictly associative
//! and 1 were strictly neutral for ⊗. In other words, we do not mention the isos
//! λ, ρ and α in our computations, just as if they were identities (see Section 2.2).
//! Given an object X and n ∈ N of L, we use X ⊗n for the nth tensor power of X:
//! X ⊗0 = 1 and X ⊗(n+1) = X ⊗n ⊗ X.
//! Given an object X of L, we define a morphism ∂ X ∈ L(!X ⊗ X, !X) as the
//! following composition of morphisms
//! !X ⊗ dX
//!
//! !X ⊗ X
//!
//! !X ⊗ !X
//!
//! cX
//!
//! !X
//!
//! n
//!
//! More generally, we define ∂ X ∈ L(!X ⊗ X ⊗n , !X):
//! 0
//!
//! ∂ X = Id!X
//! n+1
//!
//! ∂X
//! Last we set
//!
//! n
//!
//! n
//!
//! = ∂ X (∂ X ⊗ X)
//!
//! n
//!
//! dX = ∂ X (wX ⊗ X ⊗n ) ∈ L(X ⊗n , !X) .
//! n
//! We define dually ∂X ∈ L(!X, !X ⊗ X) as ∂X = (!X ⊗ dX ) cX and ∂X
//! ∈
//! n+1
//! ⊗n
//! 0
//! n
//! L(!X, !X ⊗ X ) by ∂X = Id!X and ∂X = (∂X ⊗ X) ∂X . And we set
//! n
//! dnX = (wX ⊗ X ⊗n ) ∂X
//! ∈ L(!X, X ⊗n ) .
//! 1
//!
//! Observe that we have dX = dX and d1X = dX .
//! Consider now some f ∈ L(!X, Y ), to be intuitively seen as a non linear
//! map from X to Y (if ! were assumed to be a comonad as in Section 2.6, then
//! f would be a morphism in the Kleisli category L! ). Such a morphism will
//! sometimes be called a “regular function” in the sequel, but keep in mind that
//! it is not even a function in general. For instance, if X and Y are vector spaces
//! (with a topological structure, in the infinite dimensional case), such a regular
//! function could typically be a smooth or an analytic function.
//! 45
//!
//!
//! --- PAGE BREAK ---
//! With these notations, f wX ∈ L(1, Y ) should be understood as the point of
//! Y obtained by applying the regular function f to 0. Similarly, f cX ∈ L(!X ⊗
//! !X, Y ) should be understood as the regular function g : X × X → Y defined by
//! g(x1 , x2 ) = f (x1 + x2 ).
//! Dually, given y ∈ L(1, Y ) considered as a point of Y , then y wX ∈ L(!X, Y )
//! should be understood as the constant regular function which takes y as unique
//! value. If g ∈ (!X ⊗ !X, Y ), to be considered as a regular function with two
//! parameters X × X → Y , then f = g cX ∈ L(!X, Y ) should be understood as
//! the regular function X → Y given by f (x) = g(x, x). Given g ∈ L(X, Y ), to
//! be considered as a linear function from X to Y , f = g dX ∈ L(!X, Y ) is g,
//! considered now as a regular function from X to Y .
//! The basic idea of DiLL is that f 0 = f ∂ X ∈ L(!X ⊗ X, Y ) (that is f 0 ∈
//! L(!X, X ( Y ), up to linear curryfication) represents the derivative of f .
//! Remember indeed that if f : X → Y is a smooth function from a vector
//! space X to a vector space Y , the derivative of f is a function f 0 from X to the
//! space X ( Y of (continuous) linear functions from X to Y : f 0 maps x ∈ X
//! to a linear map f 0 (x) : X → Y , the differential (or Jacobian) of f at point x,
//! which maps u ∈ X to f 0 (x) · u.
//! In particular, f dX = f ∂ X (wX ⊗ X) ∈ L(X, Y ) corresponds to f 0 (0), the
//! differential of f at 0.
//! n
//! More generally, f ∂ X ∈ L(!X ⊗ X ⊗n , Y ) represents the nth derivative of f ,
//! which is a regular function from X to the space of n-linear functions from X n to
//! Y (these n linear functions are actually symmetric, a property called “Schwarz
//! Lemma” and is axiomatized here by the commutativity of the algebra structure
//! of !X).
//! The axioms of an exponential structure express that this categorical definition of differentiation satisfies the usual laws of differentiation. Let us give
//! a simple example. Consider f ∈ L(!X ⊗ !X, Y ), to be seen as a regular function f (x1 , x2 ) depending on two parameters x1 and x2 in X. Remember that
//! g = f cX ∈ L(!X, Y ) represents the map depending on one parameter x in X
//! given by g(x) = f (x, x).
//! Using the axioms of exponential structures, one checks easily that
//! cX ∂ X = ((!X ⊗ ∂ X ) + (∂ X ⊗ !X) (!X ⊗ σ)) (cX ⊗ X)
//! which, by composing with f and using standard algebraic notations, gives
//! g 0 (x) · u = f10 (x, x) · u + f20 (x, x) · u
//! where we use fi0 for the ith partial derivative of f and “·” for the linear application of the differential. This is Leibniz law.
//! Similarly, the easily proven equation wX ∂ X = 0 expresses that the derivative
//! of a constant map is equal to 0.
//! It is a nice exercise to interpret similarly the dual equations
//! ∂X cX = (cX ⊗ X) ((!X ⊗ ∂X ) + (!X ⊗ σ) (∂X ⊗ !X))
//! ∂X wX = 0
//! 46
//!
//!
//! --- PAGE BREAK ---
//! Consider g ∈ L(!X ⊗ X, Y ), to be considered as a regular function X × X → Y
//! which is linear in its second parameter. Then f = g ∂X ∈ L(!X, Y ) is the regular
//! function X → Y given by g(x) = f (x, x). The second equation corresponds to
//! the fact that g(0, 0) = 0, and the first one, to the fact that g(x1 + x2 , x1 + x2 ) =
//! g(x1 + x2 , x1 ) + g(x1 + x2 , x2 ).
//!
//! 3.1
//!
//! Taylor exponential structures.
//!
//! Let L be an exponential structure and let f ∈ L(!X, Y ), to be considered as a
//! “regular function” from X to Y . The condition f ∂ X = 0 means intuitively that
//! the derivative of f is uniformly equal to 0, and hence, according to standard
//! intuitions on differentiation, f should be a constant map. In other words we
//! should have f = f wX wX .
//! This property can be stated in a more general way as follows: let f1 , f2 :
//! !X → Y , then
//! f1 ∂ X = f2 ∂ X ⇒ f1 + (f2 wX wX ) = f2 + (f1 wX wX )
//! and does not seem to be derivable from the other axioms of exponential structures. The converse implication is easy to prove.
//! Remark : There is a dual condition which reads as follows: if f1 , f2 : Y → !X,
//! then
//! ∂X f1 = ∂X f2 ⇒ f1 + (wX wX f2 ) = f2 + (wX wX f1 )
//! The intuition is that, given f : Y → !X, to be considered as a generalized
//! point of !X, if ∂X f = 0, then the range of f is included in the subspace of !X
//! generated by the unit of the bialgebra !X. In other words, this condition means
//! that the kernel of ∂X is generated by this unit.
//! We say that the exponential structure L is Taylor if it satisfies the first
//! condition.
//! For n ∈ N, let TnX ∈ C(!X, !X) be defined by
//! TnX =
//!
//! n
//! X
//! 1
//! i=0
//!
//! i
//!
//! i!
//!
//! dX diX .
//!
//! Lemma 9 For any n > 0, we have
//! TnX ∂ X = ∂ X (Tn−1
//! ⊗ IdX ) .
//! X
//! Proof.
//! This results from
//! n
//!
//! n−1
//!
//! dX dnX ∂ X = n∂ X ((dX
//!
//! dn−1
//! X ) ⊗ IdX )
//!
//! which comes from the basic equations of exponential structures.
//!
//! 2
//!
//! Remember that a commutative monoid M is cancellative if, in M , one has
//! u + v = u0 + v ⇒ u = u0 .
//! 47
//!
//!
//! --- PAGE BREAK ---
//! Proposition 10 Assume that L is Taylor and that each homset L(X, Y ) is a
//! n+1
//! n+1
//! cancellative monoid. Let n ∈ N and let f1 , f2 : !X → Y . If f1 ∂ X = f2 ∂ X
//! then f1 + (f2 TnX ) = f2 + (f1 TnX ).
//! n+1
//!
//! In particular, if f ∂ X = 0 (that is, the (n + 1)-th derivative of f is uniformly
//! equal to 0), then f = f TnX , meaning that f is equal to its Taylor expansion of
//! rank n.
//! Proof.
//! By induction on n. For n = 0, this is simply the hypothesis that L is Taylor.
//! n+2
//! n+2
//! Assume now that f1 ∂ X = f2 ∂ X and let us prove that f1 + (f2 Tn+1
//! X ) =
//! n+1
//! f2 + (f1 TX ).
//! n+1
//! n+1
//! We have f1 ∂ X (∂ X ⊗ IdX ) = f2 ∂ X (∂ X ⊗ IdX ). By monoidal closeness,
//! n+1
//! n+1
//! we have cur(f1 ∂ X ) ∂ X = cur(f2 ∂ X ) ∂ X and hence, by inductive hypothesis,
//! we have
//! cur(f1 ∂ X ) + (cur(f2 ∂ X ) TnX ) = cur(f2 ∂ X ) + (cur(f1 ∂ X ) TnX )
//! that is
//! cur(f1 ∂ X ) + (cur(f2 ∂ X (TnX ⊗ IdX ))) = cur(f2 ∂ X ) + (cur(f1 ∂ X (TnX ⊗ IdX )))
//! and hence
//! (f1 ∂ X ) + (f2 ∂ X (TnX ⊗ IdX )) = (f2 ∂ X ) + (f1 ∂ X (TnX ⊗ IdX ))
//! so applying Lemma 9, we get
//! n+1
//! (f1 + f2 Tn+1
//! X ) ∂ X = (f2 + f1 TX ) ∂ X .
//!
//! Applying the hypothesis that L is Taylor, we get
//! n+1
//! + (f1 + f2 Tn+1
//! f1 + f2 Tn+1
//! + (f2 + f1 Tn+1
//! X
//! X ) wX wX = f2 + f1 TX
//! X ) w X wX
//!
//! and since Tn+1
//! wX wX = wX wX we get f1 + f2 Tn+1
//! + (f1 + f2 ) wX wX =
//! X
//! X
//! n+1
//! f2 + f1 TX + (f2 + f1 ) wX wX and so, applying the cancellativeness hypothesis,
//! we get finally
//! f1 + f2 Tn+1
//! = f2 + f1 Tn+1
//! X
//! X
//! 2
//!
//! as required.
//! 3.1.1
//!
//! The category of polynomials.
//! n+1
//!
//! We say that f ∈ L(!X, Y ) is polynomial if there exists n ∈ N such that f ∂ X =
//! 0, and we call degree of f the least such n. The morphism dX ∈ C(!X, X) is
//! polynomial of degree 1.
//! 48
//!
//!
//! --- PAGE BREAK ---
//! Let f ∈ L(!X, Y ) and g ∈ L(!Y, Z) be polynomial of degree m and n respectively. We define the composition g ◦ f ∈ L(!X, Z) as follows
//! g◦f =
//!
//! n
//! X
//! 1
//!
//! i!
//! i=0
//!
//! i
//!
//! g dY f ⊗i ciX .
//!
//! i×
//!
//! }|
//! {
//! z
//! where f = f ⊗ · · · ⊗ f .
//! i
//! Since dX dX = 0 for i 6= 1, we get dX ◦ g = g. Next observe that g ◦ dX =
//! n
//! g TX = g by Proposition 10. One can prove that g ◦ f is polynomial of degree
//! mn
//! ≤ mn, that is (g ◦ f ) ∂ X = 0 by a straightforward (though boring) categorical
//! computation using the basic axioms of exponential structures. Using the same
//! axioms, one shows that this notion of composition is associative, so that we have
//! defined a category of polynomial morphisms.
//! ⊗i
//!
//! 3.1.2
//!
//! Weak functoriality and the category of polynomials.
//!
//! We do not require this operation X 7→ !X to be functorial, but some weak
//! form of functoriality can be derived from the above categorical axioms. Let
//! f ∈ L(X, Y ). By induction on n, we define a family of morphisms f n : !X → !X
//! as follows: f 0 = wX wX and
//! f n+1 = ∂ X (f n ⊗ f ) ∂X .
//! Proposition 11 Let f ∈ L(X, Y ) and g ∈ L(Y, Z) and let n, p ∈ N. Then
//! (
//! n!(g f )n if n = p
//! p n
//! g f =
//! 0
//! otherwise.
//! Proof.
//! Simple calculation using the diagram commutations which define an exponential
//! structure.
//! 2
//! Pn 1 q
//! So for each n ∈ N we can define !n f = q=0 q! f : !X → !X, and we have
//! !n g !n f = !n (g f ). So f 7→ !n f is a quasifunctor, but not a functor as it does not
//! map IdX to Id!X , but to an idempotent morphism ρnX : !X → !X.
//! In some concrete models, this sequence (!n f )n∈N can be said to be convergent, in a sense which depends of course on the model. The limit is then denoted
//! as !f and the operation defined in that way turns out often to be a true functor,
//! defining a functorial exponential in the sense of Section 2.6.
//!
//! 3.2
//!
//! Computing antiderivatives.
//!
//! We say that an exponential structure L has antiderivatives if the morphism
//! JX = IdX +(∂ X ∂X ) : !X → !X
//! 49
//!
//!
//! --- PAGE BREAK ---
//! is an isomorphism. We explain why.
//! We assume to be given an exponential structure L which has antiderivatives
//! in that sense and we set IX = JX −1 .
//! In the sequel, we use the following notation
//! ψX = (∂ X ⊗ IdX ) σ23 (∂X ⊗ IdX ) : !X ⊗ X → !X ⊗ X
//! where σ23 = !X ⊗ σ is an automorphism on !X ⊗ X ⊗ X, because this morphism
//! ψX will show up quite often. Observe in particular that
//! ∂X ∂ X = Id!X⊗X +ψX .
//! Lemma 12 The following commutation holds
//! (IX ⊗ IdX ) ψX = ψX (IX ⊗ IdX )
//!
//! (11)
//!
//! Proof.
//! −1
//!
//! Since IX = (Id!X +(∂ X ∂X )) , we have IX ⊗IdX = ϕ−1 where ϕ = Id!X⊗X +((∂ X ∂X )⊗
//! IdX ) by functoriality of ⊗. To prove (11), it suffices therefore to prove that ϕ
//! commutes with ψX . For this, it suffices to show that
//! ((∂ X ∂X ) ⊗ IdX ) ψX = ψX ((∂ X ∂X ) ⊗ IdX ) .
//! We have
//! ((∂ X ∂X ) ⊗ IdX ) ψX = ((∂ X ∂X ∂ X ) ⊗ IdX ) σ23 (∂X ⊗ IdX )
//! but remember that ∂X ∂ X = Id!X⊗X +ψX , and hence
//! ((∂ X ∂X ) ⊗ IdX ) ψX = ψX + ((∂ X ψX ) ⊗ IdX ) σ23 (∂X ⊗ IdX )
//! But ∂ X (∂ X ⊗ IdX ) σ23 = ∂ X (∂ X ⊗ IdX ) by commutativity of the bialgebra !X
//! and by definition of ∂ X . Therefore ∂ X ψX = ∂ X ((∂ X ∂X ) ⊗ IdX ). So we can
//! write
//! ((∂ X ∂X ) ⊗ IdX ) ψX
//! = ψX + (∂ X ⊗ IdX ) ((∂ X ∂X ) ⊗ IdX ⊗ IdX ) σ23 (∂X ⊗ IdX )
//! = ψX + (∂ X ⊗ IdX ) ((∂ X ∂X ) ⊗ σ) (∂X ⊗ IdX )
//! A similar, and completely symmetric computation, using this time the cocommutativity of the bialgebra !X, leads to
//! ψX ((∂ X ∂X ) ⊗ IdX ) = ψX + (∂ X ⊗ IdX ) ((∂ X ∂X ) ⊗ σ) (∂X ⊗ IdX )
//! 2
//!
//! and we are done.
//!
//! We can now prove a completely categorical version of the following proposition which is the key step in the usual proof of Poincaré’s Lemma.
//! 50
//!
//!
//! --- PAGE BREAK ---
//! Proposition 13 Let f : !X ⊗X → Y be such that the differential f (∂X ⊗IdX ) :
//! !X ⊗ X ⊗ X → Y satisfies
//! f (∂X ⊗ IdX ) σ23 = f (∂X ⊗ IdX ) .
//! Then there exists g : !X → Y such that g ∂ X = f ; in other words, g is an
//! “antiderivative” of f .
//! Proof.
//! One sets
//! g = f (IX ⊗ IdX ) ∂X .
//!
//! (12)
//!
//! Then we have
//! g ∂ X = f (IX ⊗ IdX ) ∂X ∂ X
//! = f (IX ⊗ IdX ) (Id!X⊗X +ψX )
//! = f (IX ⊗ IdX ) + f (IX ⊗ IdX ) ψX
//! = f (IX ⊗ IdX ) + f ψX (IX ⊗ IdX )
//! by Lemma 12. But
//! f ψX = f (∂ X ⊗ IdX ) σ23 (∂X ⊗ IdX )
//! = f (∂ X ⊗ IdX ) (∂X ⊗ IdX )
//!
//! by our hypothesis on f
//!
//! = f ((∂ X ∂X ) ⊗ IdX ) .
//! So we get
//! g ∂ X = f (IX ⊗ IdX ) + f ((∂ X ∂X IX ) ⊗ IdX )
//! = f (((Id!X +(∂ X ∂X )) IX ) ⊗ IdX )
//! =f
//! 2
//!
//! since IX is the inverse of Id!X +(∂ X ∂X ).
//! 3.2.1
//!
//! Comments.
//!
//! Let us give some intuition about our axiom that JX has an inverse. Given
//! f : !X → Y seen as a “regular function” from X to Y , we explain why the
//! morphism f IX : !X → Y should be understood as representing the regular
//! function g defined by
//! Z
//! 1
//!
//! g(x) =
//!
//! f (tx) dt
//! 0
//!
//! assuming of course that this integral makes sense. With this interpretation,
//! g ∂ X : !X ⊗ X → Y represents the differential Dg of g, a regular function
//! X ×X → Y which maps (x, y) to Dg(x)·y and is linear in y. Then, applying the
//! 51
//!
//!
//! --- PAGE BREAK ---
//! ordinary rules of differential calculus, and the fact that differentiation commutes
//! with integration, we get
//! Z 1
//! t(Df (tx) · y)dt .
//! Dg(x) · y =
//! 0
//!
//! The morphism h = g ∂ X ∂X : !X → Y corresponds to the regular function from
//! X to Y such that
//! Z 1
//! h(x) =
//! (Df (tx) · (tx))dt
//! 0
//!
//! Z 1
//! t(Df (tx) · x)dt
//!
//! =
//! 0
//!
//! Z 1
//!
//! df (tx)
//! dt
//! dt
//! 0
//! Z 1
//! = f (x) −
//! f (tx) dt ,
//! =
//!
//! t
//!
//! 0
//!
//! integrating by parts. In other words, we have seen that
//! f IX ∂ X ∂X = f − (f IX )
//! that is
//! f IX (Id!X +(∂ X ∂X )) = f
//! this is why our first axiom on IX is that IX (Id!X +(∂ X ∂X )) = Id!X . To explain
//! why we also require (Id!X +(∂ X ∂X )) IX = Id!X , observe that
//! l = f ∂ X ∂X : !X → Y
//! corresponds to the regular function defined by l(x) = Df (x) · x and hence l IX
//! corresponds to the regular function m : X → Y given by
//! Z 1
//! m(x) =
//! (Df (tx) · (tx))dt = h(x)
//! 0
//!
//! by linearity of the differential. So we have
//! f ∂ X ∂X IX = f − (f IX )
//! that is
//! f (Id!X +(∂ X ∂X )) IX = f .
//! A remarkable and quite natural feature of this axiomatization of antiderivatives is the fact that it is actually a mere property of the exponential structure,
//! and not an additional structure: it must be such that Id!X +(∂ X ∂X ) has an
//! inverse.
//!
//! 52
//!
//!
//! --- PAGE BREAK ---
//! Remark : The definition (12) of the antiderivative g of f in the proof above
//! reads as follows, if we use this intuitive interpretation of IX :
//! Z 1
//! g(x) =
//! f (tx) · x dt
//! (13)
//! 0
//!
//! which is exactly its definition, in the standard proof of Poincaré’s Lemma. The
//! proof of Proposition 13 is a rephrasing of the standard proof, which uses an
//! integration by parts.
//! 3.2.2
//!
//! The fundamental theorem of calculus.
//!
//! This is the statement according to which one can use antiderivatives for computRb
//! ing integrals: if f, g : R → R are such that g 0 = f , then a f (t) dt = g(b) − g(a).
//! In the present setting, it boils down to a simple categorical equation.
//! Proposition 14 Let L be an exponential structure which has antiderivatives
//! and is Taylor. Then
//! ∂ X (IX ⊗ IdX ) ∂X + wX wX = Id!X .
//! Proof.
//! Let f1 = ∂ X (IX ⊗ IdX ) ∂X : !X → !X and let f2 = Id!X . We have
//! f1 ∂ X = ∂ X (IX ⊗ IdX ) ∂X ∂ X
//! = ∂ X (IX ⊗ IdX ) + ∂ X (IX ⊗ IdX ) (∂ X ⊗ IdX ) σ23 (∂X ⊗ IdX )
//! = ∂ X (IX ⊗ IdX ) + ∂ X (∂ X ⊗ IdX ) σ23 (∂X ⊗ IdX ) (IX ⊗ IdX )
//! by Lemma 12
//! = ∂ X (IX ⊗ IdX ) + ∂ X (∂ X ⊗ IdX ) (∂X ⊗ IdX ) (IX ⊗ IdX )
//! by commutativity of cocontraction
//! = ∂ X ((Id!X +(∂ X ∂X )) ⊗ IdX ) (IX ⊗ IdX )
//! = ∂X
//!
//! since IX = (Id!X +(∂ X ∂X ))
//!
//! −1
//!
//! = f2 ∂ X .
//! Since L is Taylor and since f1 wX wX = 0, we have therefore f1 + (f2 wX wX ) =
//! f2 + (f1 wX wX ), which is exactly the announced equation.
//! 2
//! Remark : We give now an intuitive interpretation of this property. Let f : !X →
//! Y , considered as a regular function from X to Y . Then f ∂ X (IX ⊗ IdX ) ∂X :
//! !X → Y represents the regular function g : X → Y given by
//! Z 1
//! Z 1
//! df (tx)
//! g(x) =
//! Df (tx) · x dt =
//! dt
//! dt
//! 0
//! 0
//! so that g(x) = f (x) − f (0) by the Fundamental Theorem of Calculus. In other
//! words g(x) + f (0) = f (x), that is f (∂ X (IX ⊗ IdX ) ∂X + wX wX ) = f .
//!
//! 53
//!
//!
//! --- PAGE BREAK ---
//! 3.3
//!
//! Computing antiderivatives in the resource calculus
//!
//! We can consider finite linear combinations of finite resource terms (see 2.7.2) as
//! polynomials, and with this respect, it seems natural to formally compute the
//! antiderivative of such a term, as one does for polynomials. This is the purpose
//! of this short section. We use ∆ for the set of simple resource terms and khSi
//! for the free k-module generated by the set S.
//! As with ordinary polynomials, we define first the antiderivative of a monomial, that is, of a simple resource term. Remember that, for ordinary one
//! 1
//! X d+1 ; the definition is
//! variable polynomials, the antiderivative of X d is d+1
//! completely similar here. Let t ∈ ∆ be a simple resource term and let x be a
//! variable. We set
//! 1
//! Ix (t) =
//! t.
//! degx t + 1
//! We extend
//! P this operation by linearity to all elements u ∈ kh∆i, that is we set
//! Ix (u) = t∈∆ ut Ix (t).
//! (d)
//! For d ∈ N, let ∆x = {t ∈ ∆ | degx t = d} be the set of all simple resource
//! (d)
//! terms of degree d in x. The elements of kh∆x i are said to be homogeneous of
//! degree d in x.
//! With these notations, we can write
//! Ix (u) =
//!
//! ∞
//! X
//! d=0
//!
//! X
//! 1
//! ut t
//! d+1
//! (d)
//! t∈∆x
//!
//! R1
//! Intuitively, Ix (u) stands for the integral 0 u(τ x) dτ which is the basic ingredient
//! in the proof above of Poincaré’s Lemma.
//! (1)
//! Let u ∈ kh∆i which is linear in the variable h, in other words u ∈ kh∆h i.
//! 0
//! Let h be a variable which does not occur free in u, we assume that
//! ∂u [h0 /h]
//! ∂u 0
//! ·h =
//! ·h
//! ∂x
//! ∂x
//! which is our symmetry hypothesis on u. In other words, for any d ∈ N, we have
//! X
//!
//! ut
//!
//! (d)
//! t∈∆x
//!
//! X
//! ∂t
//! ∂t [h0 /h]
//! · h0 =
//! ut
//! · h.
//! ∂x
//! ∂x
//! (d)
//!
//! (14)
//!
//! t∈∆x
//!
//! Mimicking (13), we set
//! v = Ix (u) [x/h]
//! and we prove that
//! ∂v
//! · h = u.
//! ∂x
//!
//! (15)
//!
//! ∂v
//! Choose h0 as above, we prove that ∂x
//! ·h0 = u [h0 /h] which of course implies (15).
//! We have, keeping in mind that h has exactly one free occurrence in each t ∈ ∆
//!
//! 54
//!
//!
//! --- PAGE BREAK ---
//! such that ut 6= 0
//! ∞
//! X
//! ∂v 0 X 1
//! ∂t [x/h] 0
//! ut
//! ·h =
//! ·h
//! ∂x
//! d+1
//! ∂x
//! (d)
//! d=0
//!
//! =
//!
//! ∞
//! X
//! d=0
//!
//! =
//!
//! ∞
//! X
//! d=0
//!
//! =
//!
//! ∞
//! X
//! d=0
//!
//! =
//!
//! ∞
//! X
//! d=0
//!
//! t∈∆x
//!
//! 
//! 
//! X
//! ∂t
//! 1
//! ut
//! · h0 [x/h] + t [h0 /h]
//! d+1
//! ∂x
//! (d)
//! t∈∆x
//!
//! 
//! 
//! 
//! X
//! ∂t [h0 /h]
//! 1
//! ut
//! · h [x/h] + t [h0 /h]
//! d+1
//! ∂x
//! (d)
//!
//! by (14)
//!
//! t∈∆x
//!
//! 
//! 
//! X
//! ∂t [h0 /h]
//! 1
//! 0
//! ut
//! · x + t [h /h]
//! d+1
//! ∂x
//! (d)
//! t∈∆x
//!
//! X
//! 1
//! ut (d t [h0 /h] + t [h0 /h])
//! d+1
//! (d)
//!
//! since
//!
//! ∂s
//! · x = d s for all s ∈ ∆(d)
//! x
//! ∂x
//!
//! t∈∆x
//!
//! 0
//!
//! = u [h /h]
//! and we are done.
//!
//! 4
//!
//! Concrete models
//!
//! We want now to give concrete examples of categorical models of DiLL.
//!
//! 4.1
//!
//! Products, coproducts and the Seely isomorphisms
//!
//! In Section 2.6, we introduced the functorial version of the exponential without
//! mentioning the Seely isomorphisms: as explained in Paragraph 2.6.5, this choice
//! is very natural when presenting the denotational interpretation of proof-nets.
//! But when describing the structure of concrete models, as we want to do now, it
//! is more natural to assume that the linear category L is cartesian and that the
//! ! comonad is equipped with a strong symmetric monoidal structure from & to
//! ⊗.
//! So we assume to be given a preadditive *-autonomous category L equipped
//! with an exponential structure (Section 2.5) where ! is a monoidal comonad
//! satisfying the conditions of Section 2.6.
//! We assume moreover that L is cartesian, with terminal object >, cartesian
//! product &, projections πi ∈ L(X1 & X2 , Xi ). Because L is preadditive, this
//! implies that > is also an initial object, and that X1 & X2 (together with suitably
//! defined injections) is also the coproduct of X1 and X2 . In other words, L is an
//! additive monoidal category.
//! We assume to be given an isomorphism m0 ∈ L(1, !>) and a natural isomorphism m2X1 ,X2 ∈ L(!X1 ⊗ !X2 , !(X1 & X2 )) which endow the functor ! with a
//! monoidal structure. This means that diagrams similar to (6), (7), (8) and (9)
//! hold.
//! 55
//!
//!
//! --- PAGE BREAK ---
//! We also require the following diagram to commute
//! m2X,Y
//!
//! !X ⊗ !Y
//!
//! !(X & Y )
//! pX&Y
//!
//! !!(X & Y )
//!
//! pX ⊗ pY
//!
//! !h!π1 , !π2 i
//! m2!X,!Y
//!
//! !!X ⊗ !!Y
//!
//! !(!X & !Y )
//!
//! Of course, there is a connection between these two monoidal structures on
//! ! . The morphism µ0 is the following composition of morphisms:
//! 1
//!
//! m0
//!
//! !>
//!
//! p>
//!
//! !!>
//!
//! !((m01 )
//!
//! −1
//!
//! )
//!
//! !1
//!
//! and µ2X,Y is
//! !X ⊗ !Y
//!
//! m2X,Y
//!
//! !(X & Y )
//!
//! pX&Y
//!
//! !!(X & Y )
//!
//! !(m2X,Y
//!
//! −1
//!
//! )
//!
//! !(!X ⊗ !Y )
//!
//! !(dX ⊗ dY )
//!
//! !(X ⊗ Y )
//!
//! The bi-algebraic structure of !X presented in Section 2.5 is also related to
//! this Seely monoidal structure.
//! For the coalgebraic part, let ∆X ∈ L(X, X & X) be the diagonal morphism
//! associated with the cartesian product of X with itself. Then we have
//! cX = m2X,X
//!
//! −1
//!
//! !∆X : !X → !X ⊗ !X
//!
//! Similarly we set
//! wX = m01 !τX
//! where τX : X → > is the unique morphism to the terminal object. The algebraic
//! part satisfies similar conditions, using the codiagonal aX : X & X → X and the
//! 0
//! morphism τX
//! : > → X.
//!
//! 4.2
//!
//! Relational semantics
//!
//! We introduce now the simplest ∗-autonomous category equipped with an exponential structure: the category of sets and relations. For this model, we assume
//! that k = {0, 1} with addition defined by 1 + 1 = 1.
//! Let Rel be the category whose objects are sets and where Rel(X, Y ) =
//! P(X × Y ), identities being the diagonal relations and composition being defined
//! as follows: if R ∈ Rel(X, Y ) and S ∈ Rel(Y, Z) then
//! S R = {(a, c) ∈ X × Z | ∃b ∈ Y (a, b) ∈ R and (b, c) ∈ S} .
//!
//! 56
//!
//!
//! --- PAGE BREAK ---
//! Let x ⊆ X, we set Rx = {b ∈ Y | ∃a ∈ x (a, b) ∈ R} ⊆ Y which is the direct
//! image of x by R. We also define R⊥ = {(b, a) ∈ Y × X | (a, b) ∈ Y } which is
//! the transpose of R. Given x ⊆ X and y 0 ⊆ Y , we have
//! (Rx) ∩ y 0 = pr2 (R ∩ (x × y 0 ))
//!
//! and
//!
//! (R⊥ y 0 ) ∩ x = pr1 (R ∩ (x × y 0 ))
//!
//! (16)
//!
//! where pr1 and pr2 are the two projections of the cartesian product in the category
//! Set of sets and functions (the ordinary cartesian product “×”).
//! Observe that an isomorphism in Rel is a relation which is a bijection.
//! The symmetric monoidal structure is given by the tensor product X ⊗ Y =
//! X × Y and the unit 1 an arbitrary singleton. The neutrality, associativity and
//! symmetry isomorphisms are defined as the obvious corresponding bijections
//! (for instance, the symmetry isomorphism σX,Y ∈ Rel(X ⊗ Y, Y ⊗ X) is given
//! by σ(a, b) = (b, a)). This symmetric monoidal category is closed, with linear
//! function space given by X ( Y = X ×Y , the natural bijection between Rel(Z ⊗
//! X, Y ) and Rel(Z, X ( Y ) being induced by the cartesian product associativity
//! isomorphism. Last, one takes for ⊥ an arbitrary singleton, and this turns Rel
//! into a ∗-autonomous category. One denotes as ? the unique element of 1 and ⊥.
//! This category is additive, with cartesian product X1 & X2 of X1 and X2
//! defined as {1} × X1 ∪ {2} × X2 with projections πi = {((i, a), a) | a ∈ Xi } (for
//! i = 1, 2), and terminal object > = ∅. Then the commutative monoid structure
//! on homsets Rel(X, Y ) is defined by 0 = ∅ and f + g = f ∪ g and the action of k
//! on morphisms is defined by 0 f = 0 and 1 f = f (there are no other possibilities).
//! Rel is also a Seely category (see Section 4.1), for a comonad ! defined as
//! follows:
//! • !X is the set of all finite multisets of elements of X;
//! • if R ∈ Rel(X, Y ), then we set !R = {([a1 , . . . , an ], [b1 , . . . , bn ]) | n ∈
//! N and ∀i (ai , bi ) ∈ R};
//! • dX ∈ Rel(!X, X) is dX = {([a], a) | a ∈ X};
//! • pX = {(m1 + · · · + mn , [m1 , . . . , mn ]) | n ∈ N and m1 , . . . , mn ∈ !X}.
//! The monoidality isomorphism m2X,Y ∈ Rel(!X ⊗ !Y , !(X & Y )) is the bijection
//! which maps ([a1 , . . . , al ], [b1 , . . . , br ]) to [(1, a1 ), . . . , (1, al ), (2, b1 ), . . . , (2, br )].
//! Last, we also provide a codereliction natural transformation dX ∈ Rel(X, !X)
//! which is simply given by dX = {(a, [a]) | a ∈ X}.
//! With these definitions, it is easy to see that cX = {(l + r, (l, r)) | l, r ∈ !X},
//! wX = {([], ?)}, cX = {((l, r), l + r) | l, r ∈ !X} and wX = {(?, [])}. The required
//! diagrams are easily seen to commute.
//! 4.2.1
//!
//! Antiderivatives.
//!
//! This exponential structure is bicommutative and can easily seen to be Taylor
//! in the sense of Section 3.1. Moreover, it has antiderivatives in the sense of
//! Section 3.2, simply because the morphism JX = IdX +(∂ X ∂X ) : !X → !X
//! 57
//!
//!
//! --- PAGE BREAK ---
//! coincides here with the identity. Indeed ∂X = {(l + [a], (l, a)) | l ∈ !X and a ∈
//! X}, ∂ X = {((l, a), l + [a]) | l ∈ !X and a ∈ X} and therefore ∂ X ∂X = {(l, l) |
//! l ∈ !X and #l > 0}.
//! Concretely, saying that a morphism f ∈ Rel(!X ⊗ X, Y ) satisfies the symmetry condition of Proposition 13 simply means that, given m ∈ Mfin (X),
//! a, a0 ∈ X and b ∈ Y , one has ((m + [a], a0 ), b) ∈ f ⇔ ((m + [a0 ], a), b) ∈ f . In
//! that case, the antiderivative g ∈ Rel(!X, Y ) given by that proposition is simply
//! g = {(m + [a], b) | ((m, a), b) ∈ f } .
//!
//! 4.3
//!
//! Finiteness spaces
//!
//! This model can be seen as an enrichment of the model of sets and relations of
//! Section 4.2. It can also be described as a category of topological vector spaces
//! and linear continuous maps. From now on, k denotes an arbitrary field which
//! is always endowed with the discrete topology.
//!
//! 4.4
//!
//! Linearly topologized vector spaces (ltvs)
//!
//! Let E be a k-vector space. A linear topology on E is a topology λ such that
//! there is a filter L of linear subspaces of E with the following property: a subset
//! U of E is λ-open iff for any x ∈ U there exists V ∈ L such that x + V ⊆ U . One
//! says that such a filter L generates the topology L. A k-ltvs is a T
//! k-vector space
//! equipped with a linear topology. Observe that E is Hausdorff iff L = {0} (for
//! some, and hence any, generating filter L); from now on we assume always that
//! this is the case.
//! Proposition 15 Let E be a k-ltvs. Any linear subspace U of E which is a
//! neighborhood of 0 is both open and closed. So E is totally disconnected (the only
//! subsets of E which are connected are the empty set and the one point sets).
//! Proof.
//! Let L be a generating filter for the topology of E. First, let x ∈ U and let V ∈ L
//! be such that V ⊆ U (such a V exists because U is a neghborhood of 0), then
//! we have x + V ⊆ U since U is a linear subspace and hence U is open. Next let
//! x ∈ E \ U . If y ∈ U ∩ (x + U ) then we have y − x ∈ U and hence x ∈ U since
//! y ∈ U and U is a linear subspace: contradiction. Therefore U ∩ (x + U ) = ∅
//! and U is closed since U is open.
//! 2
//! Any linear subspace which contains an open linear subspace is open.
//! 4.4.1
//!
//! Cauchy completeness.
//!
//! A net in E is a family (x(d))d∈D of elements of E indexed by a directed set D.
//! The net (x(d))d∈D converges to x ∈ E if, for any neighborhood U of 0, there
//! exists d ∈ D such that ∀e ∈ D e ≥ d ⇒ x(e) − x ∈ U . Because E is Hausdorff,
//! a net converges to at most one point. As usual, one can check that a subset U
//! 58
//!
//!
//! --- PAGE BREAK ---
//! of E is open iff, for any net (x(d))d∈D which converges to a point x ∈ U , there
//! exists d ∈ D such that ∀e ∈ D e ≥ d ⇒ x(e) ∈ U .
//! A net (x(d))d∈D is Cauchy if, for any neighborhood U of 0, there exists
//! d ∈ D such that ∀e, e0 ∈ D e, e0 ≥ d ⇒ x(e) − x(e0 ) ∈ U . This latter statement
//! is equivalent to ∀e ∈ D e ≥ d ⇒ x(e) − x(d) ∈ U .
//! One says that E is complete if any Cauchy net in E converges.
//! 4.4.2
//!
//! Linear boundedness.
//!
//! Let E be an ltvs and let U be an open linear subspace of E. Let πU : E → E/U
//! be the canonical projection. This map is of course linear, and its kernel is U
//! which is a neighborhood of 0. This means that, endowing E/U with the discrete
//! topology, πU is continuous. Hence the quotient topology on E/U is the discrete
//! topology.
//! We say that a subspace B of E is linearly bounded if πU (B) is finite dimensional, for every linear open subspace U of E. In other words, for any linear open
//! subspace U , there is a finite dimensional subspace A of E such that B ⊆ U + A.
//! Proposition 16 Any finite dimensional subspace of an ltvs E is linearly bounded.
//! Let B1 and B2 be subspaces of E. If B1 ⊆ B2 and B2 is linearly bounded, so is
//! B1 . If B1 and B2 are linearly bounded, so is B1 + B2 .
//! A collection of subspaces of a vector space F having these properties is called a
//! linear bornology on F .
//! An ltvs E is locally linearly bounded if it has a linear open subspace which
//! is linearly bounded.
//! 4.4.3
//!
//! Linear and multilinear maps.
//!
//! Let E1 , . . . , En and F be k-ltvs’s. An n-multilinear function f : E1 × · · · ×
//! En → F is hypocontinuous if, for any i ∈ {1, . . . , n}, any linear open subspace
//! V ⊆ F and any linearly bounded subspaces B1 ⊆ E1 ,. . . ,Bi−1 ⊆ Ei−1 , Bi+1 ⊆
//! Ei+1 ,. . . ,Bn ⊆ En , there exists an open linear subspace U ⊆ Ei such that
//! f (B1 × · · · × Bi−1 × U × Bi+1 × · · · × Bn ) ⊆ V .
//! We denote by (E1 , . . . , En ) ( F the k-vector space of all such multilinear
//! maps. Given linearly bounded subspaces B1 , . . . , Bn of E1 , . . . , En respectively
//! and given a linear open subspace V of F , we define
//! Ann(B1 , . . . , Bn , V ) = {f ∈ (E1 , . . . , En ) ( F | f (B1 × · · · × Bn ) ⊆ V } .
//! This is a linear subspace of (E1 , . . . , En ) ( F and by Proposition 16 these
//! subspaces form a filter which defines a linear topology on (E1 , . . . , En ) ( F
//! and this topology is Hausdorff. Indeed, if f ∈ (E1 , . . . , En ) ( F is 6= 0, then
//! take xi ∈ Ei such that f (x1 , . . . , xn ) 6= 0. Since F is Hausdorff, there is a linear
//! neighborhood V of 0 in F such that f (x1 , . . . , xn ) ∈
//! / V . Let Bi = kxi ; this is a
//! linearly bounded subspace of Ei and f (B1 × · · · × Bn ) 6⊆ V .
//!
//! 59
//!
//!
//! --- PAGE BREAK ---
//! In the case n = 1 (and E = E1 ), the corresponding maps f : E → F are
//! simply called linear, and they are continuous. The corresponding function space
//! is denoted as E ( F .
//! If F = k, the corresponding maps are called (multi)linear (hypo)continuous
//! forms. If furthermore n = 1 the corresponding function space is denoted as E 0
//! and is called topological dual of E.
//! Proposition 17 Let f : E1 × · · · × En → F be multilinear and hypocontinuous
//! and let Bi ⊆ Ei be linearly bounded subspaces for i = 1, . . . , n. Then f (B1 ×
//! · · · × Bn ) is a linearly bounded subspace of F .
//! Proof.
//! Let V be an open linear subspace of F . Let U1 be an open linear subspace of E1
//! such that f (U1 × B1 × · · · × Bn ) ⊆ V . Let A1 be a finite dimensional subspace of
//! E1 such that B1 ⊆ U1 +A1 , we have f (B1 ×· · ·×Bn ) ⊆ V +f (A1 ×B2 ×· · ·×Bn ).
//! Since A1 is bounded, one can find similarly a finite dimensional subspace A2 of
//! E2 such that f (A1 × B2 × · · · × Bn ) ⊆ V + f (A1 × A2 × B3 × · · · × Bn ) and hence
//! (since V + V = V ) we get f (B1 × · · · × Bn ) ⊆ V + f (A1 × A2 × B3 × · · · × Bn ).
//! Continuing this process, we find finite dimensional subspaces Ai of Ei for i =
//! 1, . . . , n such that f (B1 × · · · × Bn ) ⊆ V + f (A1 × · · · × An ) and we conclude that
//! f (B1 × · · · × Bn ) is linearly bounded since f (A1 × · · · × An ) is finite dimensional.
//! 2
//! It is tempting to think that (multi)linear continuous maps could be characterized as those which preserve linear boundedness. This cannot be the case:
//! think of a linear map f : E → F where F is finite dimensional. Such a map
//! preserves linear boundedness (any subspace of F is linearly bounded) but has
//! no reason to be continuous.
//!
//! 4.5
//!
//! Finiteness spaces and the related ltvs’s
//!
//! We restrict now our attention to particular ltvs’s which can be described in a
//! simple combinatorial way.
//! 4.5.1
//!
//! Basic definitions.
//!
//! Let I be a set. Given F ⊆ P(I), we define F ⊥ ⊆ P(I) by
//! F ⊥ = {u0 ⊆ I | ∀u ∈ F u ∩ u0 is finite} .
//! We have F ⊆ G ⇒ G ⊥ ⊆ F ⊥ , F ⊆ F ⊥⊥ and therefore F ⊥⊥⊥ = F ⊥ .
//! A finiteness space is a pair X = (|X|, F(X)) where |X| is a set and F(X) ⊆
//! P(|X|) satisfies F(X) = F(X)⊥⊥ . The following properties follow easily from
//! the definition
//! • if u ⊆ |X| is finite then u ∈ F(X)
//! • if u, v ∈ F(X) then u ∪ v ∈ F(X)
//! 60
//!
//!
//! --- PAGE BREAK ---
//! • if u ⊆ v ∈ F(X), then u ∈ F(X).
//! Let us prove for instance the second statement. Let u0 ∈ F(X)⊥ , then (u ∪ v) ∩
//! u0 = (u ∩ u0 ) ∪ (v ∩ u0 ) is finite since both sets u ∩ u0 and v ∩ u0 are finite by
//! our hypothesis that u, v ∈ F(X). Since this holds for all u0 ∈ F(X)⊥ , we have
//! u ∪ v ∈ F(X)⊥⊥ = F(X).
//! A strong isomorphism 12 between two finiteness spaces X and Y is a bijection
//! ϕ : |X| → |Y | such that, for all u ⊆ |X|, one has u ∈ F(X) iff ϕ(u) ∈ F(Y ).
//! Let X be a finiteness space. We define a k-vector space khXi as the set of
//! all families x ∈ k|X| such that the set supp(x) = {a ∈ |X| | xa 6= 0} belongs to
//! F(X).
//! Given u0 ∈ F(X)⊥ , we define a linear subspace of khXi by
//! V X (u0 ) = {x ∈ khXi | supp(x) ∩ u0 = ∅} .
//! Observe first that ∀u0 , v 0 ∈ F(X)⊥ u0 ⊆ v 0 ⇔ V X (v 0 ) ⊆ V X (u0 ).
//! Since, given u0 , v 0 ∈ F(X)⊥ , we have V X (u0 ∪ v 0 ) = V X (u0 ) ∩ V X (v 0 ), the
//! set {V X (u0 ) |Tu0 ∈ F(X)⊥ } is a filter of linear subspaces of khXi. Moreover,
//! observe that u0 ∈F(X)⊥ V X (u0 ) = {0} (because ∀a ∈ |X| {a} ∈ F(X)⊥ ), and
//! therefore this filter defines an Hausdorff linear topology on khXi, that we call
//! the canonical topology of khXi.
//! Proposition 18 For any finiteness space X, the ltvs khXi is Cauchy-complete.
//! Proof.
//! Let (x(d))d∈D be a Cauchy net in khXi. Let a ∈ |X|. By taking u0 = {a} in
//! the definition of a Cauchy net, we see that there exist xa ∈ k and da ∈ D such
//! that ∀e ≥ da x(e)a = xa . In that way we have defined x = (xa )a∈|X| ∈ k|X|
//! We prove first that
//! ∀u0 ∈ F(X)⊥ ∃d ∈ D ∀e ≥ d ∀a ∈ u0
//!
//! x(e)a = xa .
//!
//! (17)
//!
//! Let u0 ∈ F(X)⊥ . Let d0 ∈ D be such that x(e) − x(d0 ) ∈ V X (u0 ) for all e ≥ d0 .
//! Let a ∈ u0 and let da ≥ d0 be such that x(e)a = xa for all e ≥ da . Let e ≥ d0 .
//! Let e0 ≥ e, da . We have xa = x(e0 )a since e0 ≥ da and x(e0 )a = x(e)a since
//! e, e0 ≥ d0 and a ∈ u0 . It follows that ∀a ∈ u0 xa = x(e)a .
//! From this we deduce now that x ∈ khXi. Let u0 ∈ F(X)⊥ . Let d ∈ D be
//! such that ∀e ≥ d ∀a ∈ u0 x(e)a = xa . Then supp(x) ∩ u0 = supp(x(d)) ∩ u0 is
//! finite, so supp(x) ∈ F(X)⊥⊥ = F(X), that is x ∈ khXi.
//! Now Condition (17) expresses exactly that limd∈D x(d) = x and hence the
//! net (x(d))d∈D converges.
//! 2
//! 12 This would coincide with the categorical notion of isomorphism if we were using morphisms
//!
//! which are defined as relations. With linear continuous maps (between the associated ltvs’s) as
//! morphisms, the present notion of isomorphism is a particular case of the standard categorical
//! one: we can have more linear homeomorphisms from khXi to khY i than those which are
//! generated by such finiteness-preserving bijections between webs.
//!
//! 61
//!
//!
//! --- PAGE BREAK ---
//! A natural question is whether the ltvs khXi, which is Hausdorff, is always
//! metrizable. We provide a necessary and sufficient condition under which this is
//! the case.
//! Proposition 19 Let X be a finiteness space. The ltvs khXi is metrizable iff
//! there exists a sequence (u0n )n∈N of elements of F(X)⊥ which is monotone (n ≤
//! m ⇒ u0n ⊆ u0m ) and such that ∀u0 ∈ F(X)⊥ ∃n ∈ N u0 ⊆ u0n .
//! Proof.
//! Let first (u0n )n∈N be a sequence of elements of F(X)⊥ which satisfies the condition stated above. Given x, y ∈ khXi, we define
//! 
//! 
//! if x = y
//! 0
//! d(x, y) = 2−n if x 6= y and n is the least integer
//! 
//! 
//! such that u0n ∩ supp(x − y) 6= ∅ .
//! Indeed, if x 6= y, then supp(x − y) 6= ∅ and hence, taking a ∈ supp(x − y), we
//! can find n ∈ N such that {a} ⊆ u0n . This function d is easily seen to be an
//! ultrametric distance (that is d(x, z) ≤ max(d(x, y), d(y, z))) and it generates
//! the canonical topology of khXi. Indeed we have
//! d(x, y) < 2−n
//!
//! iff x − y ∈ V X (u0n )
//!
//! (indeed, d(x, y) < 2−n means that the least m such that x−y ∈
//! / V X (u0m ) satisfies
//! 0
//! m > n) and hence B2−n = V X (un ), where Bε is the open ball centered at 0 and
//! of radius ε.
//! Conversely, assume that khXi is metrizable and let d be a distance defining
//! the canonical topology of khXi. For each n ∈ N, B2−n is a neighborhood of 0 and
//! hence there exist vn0 ∈ F(X)⊥ such that V X (vn0 ) ⊆ B2−n . Let u0n = v00 ∪· · ·∪vn0 ∈
//! F(X)⊥ . Then V X (u0n ) ⊆ V X (vn0 ) ⊆ B2−n . Now let u0 ∈ F(X)⊥ , then V X (u0 ) is
//! a neighborhood of 0 and hence there exists n such that B2−n ⊆ V X (u0 ), which
//! 2
//! implies V X (u0n ) ⊆ V X (u0 ) and hence u0 ⊆ u0n .
//! It follows that there are non metrizable ltvs associated with finiteness spaces.
//! We give in Proposition 20 an example of this situation which arises in the semantics of LL, using exponential constructions that will be introduced in Section 4.5.3.
//! Proposition 20 The ltvs kh!?1i is not metrizable
//! Proof.
//! Let X = !?1, so that |X| = Mfin (N) and a subset u for |X| belongs to F(X)
//! iff ∃n ∈ N u ⊆ Mfin ({0, . . . , n}). The proof is a typical Cantor diagonal reasoning. We assume towards a contradiction that khXi is metrizable, that is
//! by Proposition 19, we assume that there is a monotone sequence (u0n )n∈N of
//!
//! 62
//!
//!
//! --- PAGE BREAK ---
//! elements of F(X)⊥ such that ∀u0 ∈ F(X)⊥ ∃n ∈ N u0 ⊆ u0n . Let n ∈ N, we
//! have {p[n] | p ∈ N} ∈ Mfin ({0, . . . , n}) and hence u0n ∩ {p[n] | p ∈ N} is finite.
//! Therefore we can find a function f : N → N such that ∀n ∈ N f (n)[n] ∈
//! / u0n .
//! 0
//! 0
//! ⊥
//! Let u = {f (n)[n] | n ∈ N}. Then u ∈ F(X) since, for any n ∈ N,
//! u0 ∩ Mfin ({0, . . . , n}) = {f (i)[i] | i ∈ [0, n]} is finite. But for all n ∈ N we
//! 2
//! have f (n)[n] ∈ u0 \ u0n and so u0 6⊆ u0n .
//! We consider this as a very interesting phenomenon which seems to reveal a
//! relation between the topological complexity of the interpretation of a type with
//! its logical complexity (alternation of exponentials).
//! 4.5.2
//!
//! Linearly bounded subspaces.
//!
//! Let X be a finiteness space. We are interested in characterizing the linearly
//! bounded subspaces of khXi.
//! Given u ⊆ |X|, let BX (u) = {x ∈ khXi | supp(x) ⊆ u}. This is a linear
//! subspace of khXi.
//! Let u ∈ F(X). We prove that BX (u) is linearly bounded. Let u0 in F(X)⊥ .
//! Observe that V X (u0 ) = BX (|X| \ u0 ). We have therefore BX (u) ⊆ V X (u0 ) +
//! BX (u ∩ u0 ), and since u ∩ u0 is finite, the space BX (u ∩ u0 ) is finite dimensional.
//! Let U be an open subspace of U , let u0 ⊆ F(X)⊥ be such that V X (u0 ) ⊆ U .
//! Then BX (u) ⊆ U + BX (u ∩ u0 ). Hence BX (u) is linearly bounded. We show now
//! that this condition is actually sufficient.
//! Proposition 21 A linear subspace B of khXi is linearly bounded iff there exists
//! u ∈ F(X) such that B ⊆ BX (u).
//! Proof.
//! S
//! Assume that B is linearly bounded. Let u = x∈B supp(x), so that B ⊆ BX (u),
//! we prove that u ∈ F(X). Let u0 ∈ F(X)⊥ . Let A be a finite dimensional
//! subspace of E suchSthat B ⊂ V X (u0 ) + A. Let A0 be a finite generating subset
//! of A and let u0 = y∈A0 supp(y) ∈ F(X). Then x ∈ A ⇒ supp(x) ⊆ u0 (that is
//! A ⊆ BX (u0 )).
//! Let x ∈ B, we write x = x1 + x2 where x1 ∈ V X (u0 ) and x2 ∈ A. We have
//! supp(x) ⊆ supp(x1 ) ∪ supp(x2 ) and hence u0 ∩ supp(x) ⊆ (u0 ∩ supp(x1 )) ∪ (u0 ∩
//! supp(x2 )) ⊆ u0 ∩ u0 since u0 ∩ supp(x1 ) = ∅. Since this holds for all x ∈ B, we
//! have u0 ∩ u ⊆ u0 ∩ u0 so u0 ∩ u is finite and hence u ∈ F(X)
//! 2
//! Proposition 22 The ltvs khXi is locally linearly bounded iff there exist u ∈
//! F(X) and u0 ∈ F(X)⊥ such that u ∪ u0 = |X|.
//! This is an obvious consequence of Proposition 21.
//! Fink is the category whose objects are the finiteness spaces and such that
//! Fink (X, Y ) is the set of all continuous linear maps khXi → khY i.
//!
//! 63
//!
//!
//! --- PAGE BREAK ---
//! 4.5.3
//!
//! Constructions of finiteness spaces.
//!
//! We give a number of constructions on finiteness spaces which allow one to interpret differential LL, starting with the most important one, which is the linear
//! function space.
//! The most striking features of these constructions can be summarized by the
//! two following statements.
//! • In spite of the fact that these constructions are algebraic in nature (tensor
//! product, linear function space, topological dual etc), they are entirely
//! performed on the webs of the finiteness spaces and do not involve the
//! scalar coefficients. This means in particular that they do not depend on
//! the choice of the field, and this is quite surprising.
//! • So, these constructions are performed on the webs, but they do not really
//! depend on them, in the following sense. Defining an intrinsic finiteness
//! space as a k-ltvs which is linearly homeomorphic to khXi for some finiteness space X, all these constructions can be transferred to the category of
//! intrinsic finiteness spaces and continuous and linear maps.
//! Let X and Y be finiteness spaces. Let X ( Y be the finiteness space such
//! that |X ( Y | = |X| × |Y | and
//! F(X ( Y ) = {u × v 0 | u ∈ F(X) and v 0 ∈ F(Y ⊥ )}⊥
//! = {w ⊆ |X| × |Y | | ∀u ∈ F(X) ∀v 0 ∈ F(Y )⊥ w ∩ (u × v 0 ) is finite}
//! Let w ∈ F(X ( Y ), u ∈ F(X) and v 0 ∈ F(X)⊥ . It follows from (16) that
//! wu ∈ F(Y ) and that w⊥ v 0 ∈ F(X)⊥ .
//! Let M ∈ khX (PY i. If x ∈ khXi and b ∈ |Y |, then supp(M )⊥ {b} ∈ F(X)⊥
//! and hence the sum a∈|X| Ma,b xa is finite. Therefore we can define M x ∈ k|Y |
//! P
//! by M x = ( a∈|X| Ma,b xa )b∈|Y | . Since supp(M x) ⊆ supp(M ) supp(x), we have
//! M x ∈ khY i and hence the function fun(M ) defined by fun(M )(x) = M x is
//! a linear map khXi → khY i. Moreover, fun(M ) is continuous. Indeed, for
//! −1
//! any v 0 ∈ F(Y )⊥ we have V X (supp(M )⊥ v 0 ) ⊆ fun(M ) (V Y (v 0 )) and hence
//! −1
//! 0
//! ⊥ 0
//! ⊥
//! fun(M ) (V Y (v )) is open since supp(M ) v ∈ F(X) .
//! Given finiteness spaces Z1 , Z2 , we define immediately the finiteness space
//! Z1 ⊗ Z2 as Z1 ⊗ Z2 = (Z1 ( Z2⊥ )⊥ , so that |Z1 ⊗ Z2 | = |Z1 | × |Z2 |. One
//! of the most pleasant features of the theory of finiteness spaces is the following
//! property (see [Ehr05]) which has been considerably generalized in [TV10].
//! Proposition 23 Let w ⊆ |Z1 | × |Z2 |. One has w ∈ F(Z1 ⊗ Z2 ) iff pri (w) ∈
//! F(Zi ) for i = 1, 2.
//! Coming back to linear function spaces, this means in particular that, given
//! w ⊆ |X| × |Y |, one has w ∈ F(X ( Y )⊥ iff there are u ∈ F(X) and v 0 ∈ F(Y )⊥
//! such that w ⊆ u × v 0 , from which we derive a simple characterization of the
//! topology of linear function spaces.
//!
//! 64
//!
//!
//! --- PAGE BREAK ---
//! Proposition 24 The function FunX,Y : M 7→ fun(M ) is a linear homeomorphism from khX ( Y i to khXi ( khY i, equipped with the topology of uniform
//! convergence on linearly bounded subspaces.
//! Proof.
//! The proof that FunX,Y is a linear isomorphism can be found in [Ehr05]. We
//! prove that this linear isomorphism is an homeomorphism. Let B ⊆ khXi be
//! a bounded subspace and V ⊆ khY i is an open subspace. Let u ∈ F(X) be
//! such that B ⊆ BX (u) and let v 0 ∈ F(Y )⊥ be such that V Y (v 0 ) ⊆ V . Then
//! u×v 0 ∈ F(X ( Y )⊥ and hence V X(Y (u×v 0 ) ⊆ khX ( Y i is an open subspace.
//! Let M ∈ V X(Y (u×v 0 ), x ∈ B and b ∈ v 0 , we have (M x)b = 0 since supp(x) ⊆ u,
//! which shows that FunX,Y (M )(B) ⊆ V and hence FunX,Y is continuous.
//! Let now W ⊆ khX ( Y i be an open subspace. Let w ∈ F(X ( Y )⊥
//! be such that V X(Y (w) ⊆ W . By Proposition 23, there are u ∈ F(X) and
//! v 0 ∈ F(Y )⊥ such that w ⊆ u × v 0 , and hence V X(Y (u × v 0 ) ⊆ W . Then, given
//! M ∈ V X(Y (u × v 0 ), we have FunX,Y (M )(BX (u)) ⊆ V Y (v 0 ), which shows that
//! FunX,Y (W ) is an open linear subspace of khXi ( khY i.
//! We have seen that FunX,Y is a continuous and open bijection and hence it
//! is an homeomorphism.
//! 2
//! The tensor product X ⊗ Y defined above is characterized by a standard
//! universal property: it classifies the hypocontinuous bilinear maps.
//! Given vectors x ∈ khXi and y ∈ khY i, then x ⊗ y ∈ k|X⊗Y | defined by (x ⊗
//! y)(a,b) = xa yb is clearly an element of khX ⊗ Y i since supp(x ⊗ y) = supp(x) ×
//! supp(y) ∈ F(X ⊗ Y ). The map
//! τ : khXi × khY i → khX ⊗ Y i
//! (x, y) 7→ x ⊗ y
//! is obviously bilinear, let us check that it is hypocontinuous.
//! Let W be an open linear subspace of khX ⊗ Y i and let w0 ∈ F(X ⊗ Y )⊥
//! be such that V X⊗Y (w0 ) ⊆ W . Let B ⊆ khXi be a linearly bounded subspace
//! and let u ∈ F(X) be such that B ⊆ BX (u). Since w0 ∈ F(X ( Y ⊥ ) we have
//! v 0 = w0 u ∈ F(Y )⊥ . Let x ∈ B and y ∈ V Y (v 0 ), we have supp(x) ⊆ u and
//! hence pr2 (supp(x ⊗ y) ∩ w0 ) ⊆ pr2 ((u × supp(y)) ∩ w0 ) = (w0 u) ∩ supp(y) = ∅ by
//! definition of V Y (v 0 ). Therefore x ⊗ y ∈ V X⊗Y (w0 ) ⊆ W . Symmetrically, taking
//! a linearly bounded subspace C of khY i, we show that there is an open linear
//! subspace U of khXi such that τ (U × C) ⊆ W . So the map τ is bilinear and
//! hypocontinuous.
//! Proposition 25 Let Z be a finiteness space and let f : khXi × khY i → khZi
//! be bilinear and hypocontinuous. There exists exactly one continuous linear map
//! f˜ : khX ⊗ Y i → khZi such that f = f˜ ◦ τ .
//!
//! 65
//!
//!
//! --- PAGE BREAK ---
//! Proof.
//! We define a matrix M ∈ k|X|×|Y |×|Z| by Ma,b,c = f (ea , eb )c and we show first
//! that supp(M ) ∈ F(X ⊗ Y ( Z).
//! So let u ∈ F(X), v ∈ F(Y ) and w0 ∈ F(Z)⊥ ; we must show that supp(M ) ∩
//! (u × v × w0 ) is finite. Let v 0 ∈ F(Y )⊥ and u0 ∈ F(X)⊥ be such that
//! f (BX (u) × V Y (v 0 )) ⊆ V Z (w0 ) and f (V X (u0 ) × BY (v)) ⊆ V Z (w0 ) .
//! Let (a, b, c) ∈ supp(M ) ∩ (u × v × w0 ), since f (ea , eb )c ∈
//! / V Z (w0 ) (by definition
//! 0
//! of V Z (w ) and by our assumption about (a, b, c)), we must have
//! (ea , eb ) ∈
//! / BX (u) × V Y (v 0 ) and (ea , eb ) ∈
//! / V X (u0 ) × BY (v).
//! But we know that a ∈ u and b ∈ v, that is ea ∈ BX (u) and eb ∈ BY (v). It
//! follows that ea ∈
//! / V X (u0 ), that is a ∈ u0 , and similarly b ∈ v 0 .
//! Since BX (u) and BY (v) are linearly bounded, so is f (BX (u) × BY (v)) by
//! Proposition 17 and hence there exists w ∈ F(Z) such that
//! f (BX (u) × BY (v)) ⊆ BZ (w) .
//! Therefore f (ea , eb ) ∈ BZ (w) and hence c ∈ w.
//! We have shown that
//! supp(M ) ∩ (u × v × w0 ) ⊆ (u ∩ u0 ) × (v ∩ v 0 ) × (w ∩ w0 )
//! and hence supp(M ) ∩ (u × v × w0 ) is finite, so M ∈ khX ⊗ Y ( Zi.
//! Let f˜ = fun(M ), it is a linear and continuous map from khX ⊗ Y i to khZi.
//! We have f˜(ea ⊗ eb ) = f (ea , eb ) for each (a, b) ∈ |X| × |Y |. Let x ∈ F(X) and
//! y ∈ khY i, by separate continuity of f (which is a consequence of hypocontinuity)
//! we have
//! X
//! X
//! f (x, y) = f (
//! xa ea ,
//! yb eb )
//! b∈|Y |
//!
//! a∈|X|
//!
//! =
//!
//! X
//!
//! xa yb f (ea , eb )
//!
//! (a,b)∈|X|×|Y |
//!
//! =
//!
//! X
//!
//! xa yb f˜(ea ⊗ eb )
//!
//! (a,b)∈|X|×|Y |
//!
//! = f˜(x ⊗ y)
//!
//! by continuity of f˜ .
//!
//! Uniqueness of the continuous linear map f˜ results from the fact that necessarily
//! f˜(ea ⊗ eb ) = f (ea , eb ).
//! 2
//! Then one proves easily that the category Fink equipped with this tensor
//! product (whose neutral object is 1, which satisfies obviously kh1i = k) is ∗autonomous, the object of morphisms from X to Y being X ( Y and the
//! dualizing object being ⊥ = 1 (indeed, the finiteness spaces X ( ⊥ and X ⊥ are
//! obviously strongly isomorphic).
//! 66
//!
//!
//! --- PAGE BREAK ---
//! This category is preadditive in the sense of Section 2.4 since homsets Fink (X, Y )
//! have an obvious structure of k-vector space which is compatible with all the categorical operations introduced so far.
//! Countable products and coproducts are available as well. Let ˘
//! (Xi )i∈I be
//! a countable family
//! of
//! finiteness
//! spaces.
//! The
//! finiteness
//! space
//! X
//! =
//! i∈I Xi is
//! S
//! given by |X| = i∈I |Xi | and F(X) = {w ⊆ |X| | ∀i ∈ I wi ∈ F(Xi )} where
//! wi = {a ∈ |Xi | | (i, a) ∈ w}. It is easy to check that
//! F(X)⊥ = {w0 ⊆ |X| | ∀i ∈ I wi0 ∈ F(Xi )⊥ and wi0 = ∅ for almost all i}
//! ˘
//! Q
//! and it follows that F(X)⊥⊥ = F(X). It is clear that kh ˘i∈I Xi i = i∈I khXi i
//! up to a straightforward
//! strong isomorphism and that i∈I Xi together with
//! ˘
//! projections πj : kh i∈I Xi i → khXj i defined in the obvious way, is the cartesian
//! product of the Xi ’s.
//! L
//! Thanks to ∗-autonomy, the coproduct of the Xi ’s is given by
//! i∈I Xi =
//! 
//! ˘
//! L
//! Q
//! ⊥ ⊥
//! and kh i∈I Xi i ⊆ i∈I khXi i is the space of all families (xi )i∈I
//! i∈I Xi
//! of vectors such˘
//! that xi = 0 for almost all i ∈ I. Of course, the canonical linear
//! topology
//! on
//! kh
//! topology on
//! i∈I Xi i is the product topology, but the canonical
//! L
//! Q
//! kh i∈I Xi i is much finer: it is generated by all products i∈I Vi where Vi is a
//! linear neighborhood of 0 in khXi i.
//! For finite families of objects, products and coproducts coincide.
//! Let X be a finiteness space. We define !X by |!X| = Mfin (|X|) and
//! [
//! F(!X) = {A ⊆ |!X| |
//! supp(m) ∈ F(X)}
//! m∈A
//!
//! and it can be proved that indeed F(!X) = F(!X)⊥⊥ (again, see [TV10] for more
//! general results of this kind).
//! Given x ∈ khXi and m ∈ |!X|, we set
//! Y
//! xm =
//! xm(a)
//! ∈k
//! a
//! a∈|X|
//!
//! (this is a finite product since supp(m) is a finite set), so that
//! x! = (xm )m∈|!X| ∈ kh!Xi
//! by definition of F(!X). Let M ∈ kh!X ( Y i, it is not hard to see that one
//! defines a map Fun(M ) : khXi → khY i by setting
//! 
//! 
//! X
//! Fun(M )(x) = 
//! Mm,b xm 
//! m∈|!X|
//!
//! b∈|Y |
//!
//! all these sums are indeed finite, see [Ehr05] for the details. When the field k is
//! infinite, the map M 7→ Fun(M ) is injective.
//!
//! 67
//!
//!
//! --- PAGE BREAK ---
//! In [Ehr05], it is also proven that ! is a functor. Given M ∈ khX ( Y i one
//! defines !M ∈ kh!X ( !Y i by setting, for m ∈ |!X| and p ∈ |!Y |,
//! X
//! (!M )m,p =
//! [r] M r
//! r∈L(m,p)
//!
//! where
//! L(m, p) = {r ∈ Mfin (|X| × |Y |) |
//!
//! X
//!
//! X
//!
//! r(a, b) = m(a) and
//!
//! b∈|Y |
//!
//! r(a, b) = p(b)}
//!
//! a∈|X|
//!
//! (so that r ∈ L(m, p) ⇒ #m = #r = #p) and
//! [r] =
//!
//! Y
//! Q
//! b∈|Y |
//!
//! p(b)!
//! ∈ N+ .
//! a∈|X| r(a, b)!
//!
//! is a generalized multinomial coefficient.
//! This operation is functorial: !Id = Id and !M !N = !(M N ), and we also have
//! Fun(!M )(x) = !M x! = ( M x)! .
//! When k is infinite, this latter equation completely characterizes !M , by injectivity of the operation Fun in that case. This functor has a comonad structure, of
//! which we recall here only the counit dX ∈ kh!X ( Xi given by (dX )m,a = δm,[a] .
//! The bijection |!(X & Y )| → |!X ⊗ !Y | which maps the element q ∈ |!(X & Y )|
//! to the pair (m, p) ∈ |!X ⊗ !Y | defined by m(a) = q(1, a) and p(b) = q(2, b)
//! is a strong isomorphism of finiteness spaces. We also have a strong isomorphism from !⊥ to 1. These strong isomorphisms induce natural isomorphisms
//! m2X,Y ∈ Fink (!X ⊗ !Y , !(X & Y )) and m0 ∈ Fink (1, !>) which endow the functor ! with a monoidality structure from (Fink , &, >) to (Fink , ⊗, 1), satisfying
//! moreover the coherence diagram (??): to summarize, equipped with the structure described above, Fink is a Seely category, that is, a categorical model of
//! classical LL.
//! Applying the general recipe of Section 4.1, we get the contraction natural
//! transformation cX : !X → !X ⊗ !X and the weakening morphism wX : !X → 1.
//! We check that (wX )m,∗ = δm,[] and that (cX )m,(p,q) = δm,p+q . We also get the
//! cocontraction natural transformation cX : !X ⊗ !X → !X and the coweakening morphism wX : 1 → !X. And we check that (wX )∗,m = δm,[] , and that
//! 
//! (cX )(p,q),m = p+q
//! p δm,p+q where
//!  
//! Y
//! m
//! m(a)!
//! =
//! ∈ N+
//! p(a)!(m(a) − p(a))!
//! p
//! a∈|X|
//!
//! is a generalized binomial coefficient.
//! We also have a codereliction natural transformation dX : X → !X given by
//! (dX )a,m = δm,[a] , which is easily seen to satisfy the conditions of Section 2.5
//! and 2.6, so that Fink is a model of full differential LL.
//! 68
//!
//!
//! --- PAGE BREAK ---
//! 4.5.4
//!
//! An intrinsic presentation of function spaces.
//!
//! We have seen that a morphism from X to Y of the linear category Fink can be
//! seen both as an element of khX ( Y i and as a continuous linear function from
//! khXi to khY i.
//! A morphism from X to Y in the Kleisli category Relk! is an element of
//! kh!X ( Y i. Given M ∈ kh!X ( Y i, we have seen that we can define a function
//! Fun M : khXi → khY i by
//! X
//! Fun(M )(x) = M x! = (
//! Mm,b xm )b∈|Y | .
//! m∈|!X|
//!
//! Moreover, the correspondence M → Fun(M ) is functorial. We provide here an
//! intrinsic characterization of these functions.
//! Let E and F be ltvs’s. Let us say that a function f : E → F is polynomial if
//! there is n ∈ N and hypocontinuous i-linear maps fi : E i → F (for i = 0, . . . , n)
//! such that
//! f (x) = f0 + f1 (x) + · · · + fn (x, . . . , x) .
//! A polynomial map f of the form f (x) = fn (x, . . . , x), where fn is an n-linear
//! hypocontinuous function, is said to be homogeneous of degree n (this condition
//! implies of course ∀t ∈ k f (tx) = tn f (x), and when k is infinite, a polynomial
//! function is homogeneous iff it satisfies this latter condition).
//! Let Polk (E, F ) be the k-vector space of polynomial functions from E to F .
//! This space can be endowed with the linear topology of uniform convergence
//! on all linearly bounded subspaces, which admits the following generating filter base of open neighborhoods of 0: the basic opens are the linear subspaces
//! Ann(B, V ) = {f ∈ Polk (E, F ) | f (B) ⊆ V }, where B is a linearly bounded
//! subspace of E and V is a linear open subspace of F . Let Anak (E, F ) be the
//! completion13 of that ltvs.
//! Theorem 26 Assume that k is infinite. For any finiteness spaces X and Y ,
//! the ltvs kh!X ( Y i is linearly homeomorphic to Anak (khXi, khY i).
//! Proof.
//! Let h : khXin → khY i be an hypocontinuous n-linear function of matrix M ∈
//! khX ⊗ · · · ⊗ X ( Y i, so that h(x1 , . . . , xn ) = M (x1 ⊗ · · · ⊗ xn ).
//! Remember that, using contraction and dereliction, we have defined in Section 3 the morphism dnX ∈ kh!X ( X ⊗ · · · ⊗ Xi. Then we have N = M dnX ∈
//! kh!X ( Y i, and it is easy to see that
//! Fun(N )(x) = h(x, . . . , x) .
//! 13 A completion of an ltvs E is a pair (Ẽ, h) where Ẽ is a complete ltvs and h : E → Ẽ is a
//! linear and continuous map such that, for any complete ltvs F and any linear continuous map
//! f : E → F , there is an unique linear and continuous map f˜ : Ẽ → F such that f˜ ◦ h = f .
//! Using standard techniques, one can prove that any ltvs admits a completion, which is unique
//! up to unique isomorphism.
//!
//! 69
//!
//!
//! --- PAGE BREAK ---
//! In that way, we see that any polynomial map from khXi to khY i is an element
//! of kh!X ( Y i; we have an inclusion Polk (khXi, khY i) ⊆ kh!X ( Y i. Actually,
//! the exponential structure Fink is Taylor and this notion of polynomial map
//! coincides with the general notion of Section 3.1.
//! Conversely, let (m, b) ∈ |!X ( Y | with m = [a1 , . . . , an ]. The map f :
//! n
//! khXi → k defined by f (x(1), . . . , x(n)) = x(1)a1 . . . x(n)an is multilinear and
//! n
//! hypocontinuous. Hence the same holds for the map x 7→ f (x)eb from khXi
//! (|!X(Y |)
//! to khY i. Therefore we have k
//! ⊆ Polk (khXi, khY i) (given a set I, remember that k(I) is the k-vector space generated by I, that is, the space of all
//! families (ai )i∈I of elements of k such that ai = 0 for almost all i’s).
//! Hence Polk (khXi, khY i) is a dense subspace of kh!X ( Y i. To show that
//! kh!X ( Y i is the completion of Polk (khXi, khY i) it suffices to show that the
//! above defined linear topology on that space (uniform convergence on all linearly
//! bounded subspaces) is the restriction of the topology of kh!X ( Y i.
//! Let B ⊆ khXi be a linearly bounded subspace and let V ⊆ khY i be linear
//! open. Let v 0 ∈ F(Y )⊥ be such that V Y (v 0 ) ⊆ V . By Proposition 21, supp(B) ∈
//! F(X), so Mfin (supp(B)) ∈ F(!X). Let
//! M ∈ V !X(Y (Mfin (supp(B)) × v 0 ) ⊆ kh!X ( Y i ,
//! then fun(M )(x)b = 0 for each x ∈ B and b ∈ v 0 . So we have
//! V !X(Y (Mfin (supp(B)) × v 0 ) ∩ Polk (khXi, khY i) ⊆ Ann(B, V ) .
//! S
//! Conversely let U ∈ F(!X) and v 0 ∈ F(Y )⊥ , then we have u = m∈U supp(m) ∈
//! F(X) and hence the subspace B ⊆ khXi of all vectors which vanish outside u
//! is linearly bounded. Let M ∈ kh!X ( Y i be such that the map Fun(M ) :
//! khXi → khY i is polynomial and belongs to Ann(B, V Y (v 0 )). Then for any
//! m = [a1 , . . . , an ] ∈ Mfin (u) and b ∈ v 0 we have Mm,b = 0 because this
//! m(a )
//! m(a )
//! scalar is the coefficient of the monomial ξ1 1 . . . ξn n in the polynomial
//! P ∈ k[ξ1 , . . . , ξn ] such that P (z1 , . . . , zn ) = Fun(M )(x)b where x ∈ khXi is
//! such that xa = zi if a = ai and xa = 0 if a ∈
//! / supp(m), and P = 0 because
//! Fun(M )(B) ⊆ V Y (v 0 ) by assumption (we also use the fact that k is infinite).
//! Hence M ∈ V !X(Y (U × v 0 ) and we have shown that
//! Ann(B, V Y (v 0 )) ⊆ V !X(Y (U × v 0 ) ∩ Polk (khXi, khY i) ,
//! showing that this latter set is a neighborhood of 0 in the space of polynomials.
//! 2
//! The Taylor formula proved in [Ehr05] for the morphisms of this Kleisli category shows that actually any morphism is the sum of a converging series whose
//! n-th term is an homogeneous polynomial of degree n.
//! As an example, take E = k[ξ] ≃ kh!1 ( 1i. The corresponding topology on
//! E is the discrete topology. A typical example of generalized polynomial map
//! is the function ϕ : E → k which maps a polynomial P to P (P (0)), in other
//! words, ϕ(x0 + x1 ξ + · · · + xn ξ n ) = x0 + x1 x0 + · · · + xn xn0 . Considered as a
//! 70
//!
//!
//! --- PAGE BREAK ---
//! generalized polynomial of infinitely many variables x0 , x1 , . . . , we see that ϕ is
//! not of bounded degree, and so it is not polynomial. Nevertheless, it corresponds
//! to a very simple and finite computation on polynomials.
//! 4.5.5
//!
//! Antiderivatives.
//!
//! Just as Rel, the Fink exponential structure is Taylor in the sense of Section 3.1. Moreover, if k is of characteristic 0 (meaning that ∀n ∈ N n 1 =
//! 0 ⇒ n = 0) it has antiderivatives in the sense of 3.2, because the morphism
//! JX = IdX +(∂ X ∂X ) : !X → !X satisfies (JX )p,q = (#p + 1)δp,q for all p, q ∈ |!X|
//! 1
//! and hence is an isomorphism whose inverse IX is given by (IX )p,q = #p+1
//! δp,q .
//!
//! Acknowledgment
//! Part of the work reported in this article has been supported by the FrenchChinese project ANR-11-IS02-0002 and NSFC 61161130530 Locali.
//!
//! References
//! [Abr93]
//!
//! Samson Abramsky. Computational interpretations of linear logic.
//! Theoretical Computer Science, 111:3–57, 1993.
//!
//! [BCEM11] Antonio Bucciarelli, Alberto Carraro, Thomas Ehrhard, and Giulio
//! Manzonetto. Full Abstraction for Resource Calculus with Tests. In
//! Marc Bezem, editor, CSL, volume 12 of LIPIcs. Schloss Dagstuhl Leibniz-Zentrum fuer Informatik, 2011.
//! [BCL99]
//!
//! Gérard Boudol, Pierre-Louis Curien, and Carolina Lavatelli. A semantics for lambda calculi with resource. Mathematical Structures
//! in Computer Science, 9(4):437–482, 1999.
//!
//! [BCS06]
//!
//! Richard Blute, Robin Cockett, and Robert Seely. Differential categories. Mathematical Structures in Computer Science, 16(6):1049–
//! 1083, 2006.
//!
//! [BET12]
//!
//! Richard Blute, Thomas Ehrhard, and Christine Tasson. A convenient differential category. Cahiers de Topologie et Géométrie
//! Différentielle Catégoriques, 53, 2012.
//!
//! [Bie95]
//!
//! Gavin Bierman. What is a categorical model of intuitionistic linear
//! logic? In Mariangiola Dezani-Ciancaglini and Gordon D. Plotkin,
//! editors, Proceedings of the second Typed Lambda-Calculi and Applications conference, volume 902 of Lecture Notes in Computer Science, pages 73–93. Springer-Verlag, 1995.
//!
//! 71
//!
//!
//! --- PAGE BREAK ---
//! [Cur09]
//!
//! Pierre-Louis Curien, editor. Typed Lambda Calculi and Applications,
//! 9th International Conference, TLCA 2009, Brasilia, Brazil, July 13, 2009. Proceedings, volume 5608 of Lecture Notes in Computer
//! Science. Springer, 2009.
//!
//! [DB87]
//!
//! N.G. De Bruijn. Generalizing Automath by means of a lambdatyped lambda calculus. In D.W. Kueker, E.G.K. Lopez-Escobar,
//! and C.H. Smith, editors, Mathematical Logic and Theoretical Computer Science, Lecture Notes in Pure and Applied Mathematics,
//! pages 71–92. Marcel Dekker, 1987. Reprinted in: Selected papers
//! on Automath, Studies in Logic, volume 133, pages 313-337, NorthHolland, 1994.
//!
//! [DR99]
//!
//! Vincent Danos and Laurent Regnier. Reversible, irreversible and
//! optimal lambda-machines. Theoretical Computer Science, 227(12):273–291, 1999.
//!
//! [Ehr02]
//!
//! Thomas Ehrhard. On Köthe sequence spaces and linear logic. Mathematical Structures in Computer Science, 12:579–623, 2002.
//!
//! [Ehr05]
//!
//! Thomas Ehrhard. Finiteness spaces. Mathematical Structures in
//! Computer Science, 15(4):615–646, 2005.
//!
//! [Ehr10]
//!
//! Thomas Ehrhard. A finiteness structure on resource terms. In LICS,
//! pages 402–410. IEEE Computer Society, 2010.
//!
//! [Ehr12]
//!
//! Thomas Ehrhard. The Scott model of Linear Logic is the extensional collapse of its relational model. Theoretical Computer Science,
//! 424:20–45, 2012.
//!
//! [Ehr14]
//!
//! Thomas Ehrhard. A new correctness criterion for MLL proof nets.
//! In Thomas A. Henzinger and Dale Miller, editors, Joint Meeting of
//! the Twenty-Third EACSL Annual Conference on Computer Science
//! Logic (CSL) and the Twenty-Ninth Annual ACM/IEEE Symposium
//! on Logic in Computer Science (LICS), CSL-LICS ’14, Vienna, Austria, July 14 - 18, 2014, page 38. ACM, 2014.
//!
//! [ER03]
//!
//! Thomas Ehrhard and Laurent Regnier. The differential lambdacalculus. Theoretical Computer Science, 309(1-3):1–41, 2003.
//!
//! [ER06]
//!
//! Thomas Ehrhard and Laurent Regnier. Böhm trees, Krivine machine and the Taylor expansion of ordinary lambda-terms. In
//! Arnold Beckmann, Ulrich Berger, Benedikt Löwe, and John V.
//! Tucker, editors, Logical Approaches to Computational Barriers, volume 3988 of Lecture Notes in Computer Science, pages 186–197.
//! Springer-Verlag, 2006. Long version available on http://www.pps.
//! univ-paris-diderot.fr/~ehrhard/.
//!
//! 72
//!
//!
//! --- PAGE BREAK ---
//! [ER08]
//!
//! Thomas Ehrhard and Laurent Regnier. Uniformity and the Taylor
//! expansion of ordinary lambda-terms. Theoretical Computer Science,
//! 403(2-3):347–372, 2008.
//!
//! [Fio07]
//!
//! Marcelo P. Fiore. Differential structure in models of multiplicative
//! biadditive intuitionistic linear logic. In Simona Ronchi Della Rocca,
//! editor, TLCA, volume 4583 of Lecture Notes in Computer Science,
//! pages 163–177. Springer, 2007.
//!
//! [FM99]
//!
//! Maribel Fernández and Ian Mackie. A Calculus for Interaction Nets.
//! In Gopalan Nadathur, editor, PPDP, volume 1702 of Lecture Notes
//! in Computer Science, pages 170–187. Springer-Verlag, 1999.
//!
//! [Gim11]
//!
//! Stéphane Gimenez. Realizability proof for normalization of full differential linear logic. In C.-H. Luke Ong, editor, TLCA, volume
//! 6690 of Lecture Notes in Computer Science, pages 107–122. SpringerVerlag, 2011.
//!
//! [Gir86]
//!
//! Jean-Yves Girard. The system F of variable types, fifteen years
//! later. Theoretical Computer Science, 45:159–192, 1986.
//!
//! [Gir87]
//!
//! Jean-Yves Girard. Linear logic. Theoretical Computer Science, 50:1–
//! 102, 1987.
//!
//! [Gir88]
//!
//! Jean-Yves Girard. Normal functors, power series and the λ-calculus.
//! Annals of Pure and Applied Logic, 37:129–177, 1988.
//!
//! [Hut93]
//!
//! Michael Huth. Linear Domains and Linear Maps. In Stephen D.
//! Brookes, Michael G. Main, Austin Melton, Michael W. Mislove, and
//! David A. Schmidt, editors, MFPS, volume 802 of Lecture Notes in
//! Computer Science, pages 438–453. Springer-Verlag, 1993.
//!
//! [Kri85]
//!
//! Jean-Louis Krivine. Un interprteur du lambda-calcul. Unpublished
//! note, 1985.
//!
//! [Kri07]
//!
//! Jean-Louis Krivine. A call-by-name lambda-calculus machine.
//! Higher-Order and Symbolic Computation, 20(3):199–207, 2007.
//!
//! [Mac71]
//!
//! Saunders Mac Lane. Categories for the Working Mathematician,
//! volume 5 of Graduate Texts in Mathematics. Springer-Verlag, 1971.
//!
//! [Mel09]
//!
//! Paul-André Melliès. Categorical semantics of linear logic. Panoramas et Synthèses, 27, 2009.
//!
//! [MS08]
//!
//! Ian Mackie and Shinya Sato. A Calculus for Interaction Nets Based
//! on the Linear Chemical Abstract Machine. Electronic Notes in Theoretical Computer Science, 192(3):59–70, 2008.
//!
//! [Pag09]
//!
//! Michele Pagani. The cut-elimination theorem for differential nets
//! with promotion. In Curien [Cur09], pages 219–233.
//!
//! 73
//!
//!
//! --- PAGE BREAK ---
//! [PT09]
//!
//! Michele Pagani and Paolo Tranquilli. Parallel Reduction in Resource
//! Lambda-Calculus. In Zhenjiang Hu, editor, APLAS, volume 5904 of
//! Lecture Notes in Computer Science, pages 226–242. Springer, 2009.
//!
//! [PT11]
//!
//! Michele Pagani and Paolo Tranquilli. The Conservation Theorem
//! for Differential Nets. Mathematical Structures in Computer Science,
//! 2011. To appear.
//!
//! [Ret03]
//!
//! Christian Retoré. Handsome proof-nets: perfect matchings and
//! cographs. Theoretical Computer Science, 294(3):473–488, 2003.
//!
//! [Tas09a]
//!
//! Christine Tasson. Algebraic totality, towards completeness.
//! Curien [Cur09], pages 325–340.
//!
//! [Tas09b]
//!
//! Christine Tasson. Sémantiques et syntaxes vectorielles de la logique
//! linéaire. Thèse de doctorat, Université Paris Diderot – Paris 7, 2009.
//!
//! [Tra09]
//!
//! Paolo Tranquilli. Confluence of pure differential nets with promotion. In Erich Grädel and Reinhard Kahle, editors, CSL, volume
//! 5771 of Lecture Notes in Computer Science, pages 500–514. SpringerVerlag, 2009.
//!
//! [TV10]
//!
//! Christine Tasson and Lionel Vaux. Transport of finiteness structures
//! and applications. Mathematical Structures in Computer Science,
//! 2010. To appear.
//!
//! [Vau05]
//!
//! Lionel Vaux. The differential lambda-mu calculus. Theoretical Computer Science, 379(1-2):166–209, 2005.
//!
//! [Vau09]
//!
//! Lionel Vaux. The algebraic lambda-calculus. Mathematical Structures in Computer Science, 19(5):1029–1059, 2009.
//!
//! [Win04]
//!
//! Glynn Winskel. Linearity and non linearity in distributed computation. In Thomas Ehrhard, Jean-Yves Girard, Paul Ruet, and Philip
//! Scott, editors, Linear Logic in Computer Science, volume 316 of
//! London Mathematical Society Lecture Notes Series. Cambridge University Press, 2004.
//!
//! 74
//!
//! In
//!
//!
//! --- PAGE BREAK ---
//!
//! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//!
