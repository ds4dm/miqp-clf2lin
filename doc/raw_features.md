## MIQPs - Raw Features List 

0. nameID

**General**

1.   optimization sense (min vs max)
2.   number of binary variables
3.   number of integer variables
4.   total number of variables (size of the pb)
5.   total number of constraints

**Quadratic objective**

6.   number of nnz square binaries (diagonal only)
7.   number of nnz square continuous (diagonal only)
8.   number of nnz square integer (diagonal only)
9.   number of (bin, bin) out-diagonal products 
10.  number of (cont, cont) out-diagonal products 
11.  number of (int, int) out-diagonal products 
12.  number of (bin, cont) out-diagonal products 
13.  number of (bin, int) out-diagonal products 
14.  number of (int, cont) out-diagonal products 
15.  max degree of binary variables
16.  min degree of binary variables
17.  avg degree of binary variables
18.  max degree of continuous variables
19.  min degree of continuous variables
20.  avg degree of continuous variables
21.  max degree of integer variables
22.  min degree of integer variables
23.  avg degree of integer variables
24.   density of connectivity graph
25.  smallest nnz |q_ii| (or 0) (diagonal)
26.  biggest nnz |q_ii| (or 0) (diagonal)
27.  smallest nnz |q_ij| (or 0) (all)
28.  biggest nnz |q_ij| (or 0) (all)
29.  averaged 'diagonal dominance' on rows

**Linear objective**

30.  number of nnz binary in linear objective
31.  number of of nnz continuous in linear objective
32.  number of of nnz integer in linear objective
33.  min nnz c_i
34.  max nnz c_i

**Constraints**

35.  number of nnz binary in constraints
36.  number of nnz continuous in constraints
37.  number of nnz integer in constraints
38.  number of constraints involving binary variables
39.  number of constraints involving continuous variables
40.  number of constraints involving integer variables
41.  min nnz |a_ij|
42.  max nnz |a_ij|
43.  min rhs nnz
44.  max rhs nnz

**Spectrum**

45.  number of positive eigenvalues
46.  number of negative eigenvalues
47.  number of zero eigenvalues
48.  value lambda_max
49.  value lambda_min
50.  trace of Q
51.  spectral norm of Q
52.  number of positive eigenvalues (after abs_tol correction)
53.  number of negative eigenvalues (after abs_tol correction)
54.  number of zero eigenvalues (after abs_tol correction)
55.  value lambda_max (after abs_tol correction)
56.  value lambda_min (after abs_tol correction)
57.  trace of Q (after abs_tol correction)
58.  spectral norm of Q (after abs_tol correction)

**Root node** (from res logs)

59. RootDualBound_L
60. RootDualBound_NL
61. RtTime_L
62. RtTime_NL
63. RLPTime_L
64. RLPTime_NL
65. Vars_L
66. Vars_NL
67. Conss_L
68. Conss_NL
69. Nonzs_L
70. Nonzs_NL
