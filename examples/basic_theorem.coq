Require Import Arith.
Require Import Logic.

(* Definition: unnamed *)
(* \label{def:even} A natural number $n$ is called even if there exists a natural number $k$ such that $n = 2k$. *)
Definition even_number (n : nat) : Prop :=
  exists k : nat, n = 2 * k.

(* Theorem: unnamed *)
(* \label{thm:sum_even} The sum of two even numbers is even. *)
Theorem sum_of_even_numbers : forall a b : nat,
  even_number a -> even_number b -> even_number (a + b).
Proof.
  intros a b Ha Hb.
  destruct Ha as [k Hk].
  destruct Hb as [l Hl].
  exists (k + l).
  rewrite Hk, Hl.
  ring.
Qed.

(* Theorem: unnamed *)
(* If $n$ is even, then $n^2$ is even. *)
Theorem sum_of_even_numbers : forall a b : nat,
  even_number a -> even_number b -> even_number (a + b).
Proof.
  intros a b Ha Hb.
  destruct Ha as [k Hk].
  destruct Hb as [l Hl].
  exists (k + l).
  rewrite Hk, Hl.
  ring.
Qed.
