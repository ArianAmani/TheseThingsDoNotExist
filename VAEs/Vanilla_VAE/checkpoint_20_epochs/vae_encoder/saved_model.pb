Με
Φͺ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Exp
x"T
y"T"
Ttype:

2
ϋ
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ρ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
Α
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.22v2.9.1-132-g18960c44ad38νΐ
}
Dense_Log_Var/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameDense_Log_Var/bias
v
&Dense_Log_Var/bias/Read/ReadVariableOpReadVariableOpDense_Log_Var/bias*
_output_shapes	
:*
dtype0

Dense_Log_Var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameDense_Log_Var/kernel

(Dense_Log_Var/kernel/Read/ReadVariableOpReadVariableOpDense_Log_Var/kernel* 
_output_shapes
:
*
dtype0
s
Dense_MU/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameDense_MU/bias
l
!Dense_MU/bias/Read/ReadVariableOpReadVariableOpDense_MU/bias*
_output_shapes	
:*
dtype0
|
Dense_MU/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameDense_MU/kernel
u
#Dense_MU/kernel/Read/ReadVariableOpReadVariableOpDense_MU/kernel* 
_output_shapes
:
*
dtype0
}
Dense_Layer_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameDense_Layer_6/bias
v
&Dense_Layer_6/bias/Read/ReadVariableOpReadVariableOpDense_Layer_6/bias*
_output_shapes	
:*
dtype0

Dense_Layer_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
@*%
shared_nameDense_Layer_6/kernel

(Dense_Layer_6/kernel/Read/ReadVariableOpReadVariableOpDense_Layer_6/kernel* 
_output_shapes
:
@*
dtype0

BN_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameBN_5/moving_variance
z
(BN_5/moving_variance/Read/ReadVariableOpReadVariableOpBN_5/moving_variance*
_output_shapes	
:*
dtype0
y
BN_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameBN_5/moving_mean
r
$BN_5/moving_mean/Read/ReadVariableOpReadVariableOpBN_5/moving_mean*
_output_shapes	
:*
dtype0
k
	BN_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	BN_5/beta
d
BN_5/beta/Read/ReadVariableOpReadVariableOp	BN_5/beta*
_output_shapes	
:*
dtype0
m

BN_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
BN_5/gamma
f
BN_5/gamma/Read/ReadVariableOpReadVariableOp
BN_5/gamma*
_output_shapes	
:*
dtype0
s
Conv2D_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConv2D_5/bias
l
!Conv2D_5/bias/Read/ReadVariableOpReadVariableOpConv2D_5/bias*
_output_shapes	
:*
dtype0

Conv2D_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameConv2D_5/kernel
}
#Conv2D_5/kernel/Read/ReadVariableOpReadVariableOpConv2D_5/kernel*(
_output_shapes
:*
dtype0

BN_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameBN_4/moving_variance
z
(BN_4/moving_variance/Read/ReadVariableOpReadVariableOpBN_4/moving_variance*
_output_shapes	
:*
dtype0
y
BN_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameBN_4/moving_mean
r
$BN_4/moving_mean/Read/ReadVariableOpReadVariableOpBN_4/moving_mean*
_output_shapes	
:*
dtype0
k
	BN_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	BN_4/beta
d
BN_4/beta/Read/ReadVariableOpReadVariableOp	BN_4/beta*
_output_shapes	
:*
dtype0
m

BN_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
BN_4/gamma
f
BN_4/gamma/Read/ReadVariableOpReadVariableOp
BN_4/gamma*
_output_shapes	
:*
dtype0
s
Conv2D_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConv2D_4/bias
l
!Conv2D_4/bias/Read/ReadVariableOpReadVariableOpConv2D_4/bias*
_output_shapes	
:*
dtype0

Conv2D_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameConv2D_4/kernel
}
#Conv2D_4/kernel/Read/ReadVariableOpReadVariableOpConv2D_4/kernel*(
_output_shapes
:*
dtype0

BN_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameBN_3/moving_variance
z
(BN_3/moving_variance/Read/ReadVariableOpReadVariableOpBN_3/moving_variance*
_output_shapes	
:*
dtype0
y
BN_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameBN_3/moving_mean
r
$BN_3/moving_mean/Read/ReadVariableOpReadVariableOpBN_3/moving_mean*
_output_shapes	
:*
dtype0
k
	BN_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	BN_3/beta
d
BN_3/beta/Read/ReadVariableOpReadVariableOp	BN_3/beta*
_output_shapes	
:*
dtype0
m

BN_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
BN_3/gamma
f
BN_3/gamma/Read/ReadVariableOpReadVariableOp
BN_3/gamma*
_output_shapes	
:*
dtype0
s
Conv2D_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConv2D_3/bias
l
!Conv2D_3/bias/Read/ReadVariableOpReadVariableOpConv2D_3/bias*
_output_shapes	
:*
dtype0

Conv2D_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameConv2D_3/kernel
|
#Conv2D_3/kernel/Read/ReadVariableOpReadVariableOpConv2D_3/kernel*'
_output_shapes
:@*
dtype0

BN_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameBN_2/moving_variance
y
(BN_2/moving_variance/Read/ReadVariableOpReadVariableOpBN_2/moving_variance*
_output_shapes
:@*
dtype0
x
BN_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameBN_2/moving_mean
q
$BN_2/moving_mean/Read/ReadVariableOpReadVariableOpBN_2/moving_mean*
_output_shapes
:@*
dtype0
j
	BN_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	BN_2/beta
c
BN_2/beta/Read/ReadVariableOpReadVariableOp	BN_2/beta*
_output_shapes
:@*
dtype0
l

BN_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
BN_2/gamma
e
BN_2/gamma/Read/ReadVariableOpReadVariableOp
BN_2/gamma*
_output_shapes
:@*
dtype0
r
Conv2D_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameConv2D_2/bias
k
!Conv2D_2/bias/Read/ReadVariableOpReadVariableOpConv2D_2/bias*
_output_shapes
:@*
dtype0

Conv2D_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameConv2D_2/kernel
{
#Conv2D_2/kernel/Read/ReadVariableOpReadVariableOpConv2D_2/kernel*&
_output_shapes
: @*
dtype0

BN_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameBN_1/moving_variance
y
(BN_1/moving_variance/Read/ReadVariableOpReadVariableOpBN_1/moving_variance*
_output_shapes
: *
dtype0
x
BN_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameBN_1/moving_mean
q
$BN_1/moving_mean/Read/ReadVariableOpReadVariableOpBN_1/moving_mean*
_output_shapes
: *
dtype0
j
	BN_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	BN_1/beta
c
BN_1/beta/Read/ReadVariableOpReadVariableOp	BN_1/beta*
_output_shapes
: *
dtype0
l

BN_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
BN_1/gamma
e
BN_1/gamma/Read/ReadVariableOpReadVariableOp
BN_1/gamma*
_output_shapes
: *
dtype0
r
Conv2D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv2D_1/bias
k
!Conv2D_1/bias/Read/ReadVariableOpReadVariableOpConv2D_1/bias*
_output_shapes
: *
dtype0

Conv2D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameConv2D_1/kernel
{
#Conv2D_1/kernel/Read/ReadVariableOpReadVariableOpConv2D_1/kernel*&
_output_shapes
: *
dtype0

NoOpNoOp
s
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ξr
valueΔrBΑr BΊr
θ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
layer_with_weights-12
layer-14
layer-15
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
Θ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias
 !_jit_compiled_convolution_op*
Υ
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(axis
	)gamma
*beta
+moving_mean
,moving_variance*
Θ
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias
 5_jit_compiled_convolution_op*
Υ
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
<axis
	=gamma
>beta
?moving_mean
@moving_variance*
Θ
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

Gkernel
Hbias
 I_jit_compiled_convolution_op*
Υ
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance*
Θ
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

[kernel
\bias
 ]_jit_compiled_convolution_op*
Υ
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance*
Θ
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

okernel
pbias
 q_jit_compiled_convolution_op*
Υ
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
xaxis
	ygamma
zbeta
{moving_mean
|moving_variance*

}	variables
~trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses* 
 
0
 1
)2
*3
+4
,5
36
47
=8
>9
?10
@11
G12
H13
Q14
R15
S16
T17
[18
\19
e20
f21
g22
h23
o24
p25
y26
z27
{28
|29
30
31
32
33
34
35*
Π
0
 1
)2
*3
34
45
=6
>7
G8
H9
Q10
R11
[12
\13
e14
f15
o16
p17
y18
z19
20
21
22
23
24
25*
* 
΅
‘non_trainable_variables
’layers
£metrics
 €layer_regularization_losses
₯layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
¦trace_0
§trace_1
¨trace_2
©trace_3* 
:
ͺtrace_0
«trace_1
¬trace_2
­trace_3* 
* 

?serving_default* 

0
 1*

0
 1*
* 

―non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

΄trace_0* 

΅trace_0* 
_Y
VARIABLE_VALUEConv2D_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEConv2D_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
)0
*1
+2
,3*

)0
*1*
* 

Άnon_trainable_variables
·layers
Έmetrics
 Ήlayer_regularization_losses
Ίlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

»trace_0
Όtrace_1* 

½trace_0
Ύtrace_1* 
* 
YS
VARIABLE_VALUE
BN_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	BN_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEBN_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEBN_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

30
41*

30
41*
* 

Ώnon_trainable_variables
ΐlayers
Αmetrics
 Βlayer_regularization_losses
Γlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

Δtrace_0* 

Εtrace_0* 
_Y
VARIABLE_VALUEConv2D_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEConv2D_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
=0
>1
?2
@3*

=0
>1*
* 

Ζnon_trainable_variables
Ηlayers
Θmetrics
 Ιlayer_regularization_losses
Κlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

Λtrace_0
Μtrace_1* 

Νtrace_0
Ξtrace_1* 
* 
YS
VARIABLE_VALUE
BN_2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	BN_2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEBN_2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEBN_2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

G0
H1*

G0
H1*
* 

Οnon_trainable_variables
Πlayers
Ρmetrics
 ?layer_regularization_losses
Σlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

Τtrace_0* 

Υtrace_0* 
_Y
VARIABLE_VALUEConv2D_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEConv2D_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
Q0
R1
S2
T3*

Q0
R1*
* 

Φnon_trainable_variables
Χlayers
Ψmetrics
 Ωlayer_regularization_losses
Ϊlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

Ϋtrace_0
άtrace_1* 

έtrace_0
ήtrace_1* 
* 
YS
VARIABLE_VALUE
BN_3/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	BN_3/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEBN_3/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEBN_3/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

[0
\1*

[0
\1*
* 

ίnon_trainable_variables
ΰlayers
αmetrics
 βlayer_regularization_losses
γlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

δtrace_0* 

εtrace_0* 
_Y
VARIABLE_VALUEConv2D_4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEConv2D_4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
e0
f1
g2
h3*

e0
f1*
* 

ζnon_trainable_variables
ηlayers
θmetrics
 ιlayer_regularization_losses
κlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

λtrace_0
μtrace_1* 

νtrace_0
ξtrace_1* 
* 
YS
VARIABLE_VALUE
BN_4/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	BN_4/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEBN_4/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEBN_4/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

o0
p1*

o0
p1*
* 

οnon_trainable_variables
πlayers
ρmetrics
 ςlayer_regularization_losses
σlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*

τtrace_0* 

υtrace_0* 
_Y
VARIABLE_VALUEConv2D_5/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEConv2D_5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
y0
z1
{2
|3*

y0
z1*
* 

φnon_trainable_variables
χlayers
ψmetrics
 ωlayer_regularization_losses
ϊlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*

ϋtrace_0
όtrace_1* 

ύtrace_0
ώtrace_1* 
* 
YS
VARIABLE_VALUE
BN_5/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	BN_5/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEBN_5/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEBN_5/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

?non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
e_
VARIABLE_VALUEDense_Layer_6/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEDense_Layer_6/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEDense_MU/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEDense_MU/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
e_
VARIABLE_VALUEDense_Log_Var/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEDense_Log_Var/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses* 

 trace_0
‘trace_1* 

’trace_0
£trace_1* 
J
+0
,1
?2
@3
S4
T5
g6
h7
{8
|9*
z
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

+0
,1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
@1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

S0
T1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

g0
h1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

{0
|1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

serving_default_Encoder_InputPlaceholder*1
_output_shapes
:?????????*
dtype0*&
shape:?????????
ύ
StatefulPartitionedCallStatefulPartitionedCallserving_default_Encoder_InputConv2D_1/kernelConv2D_1/bias
BN_1/gamma	BN_1/betaBN_1/moving_meanBN_1/moving_varianceConv2D_2/kernelConv2D_2/bias
BN_2/gamma	BN_2/betaBN_2/moving_meanBN_2/moving_varianceConv2D_3/kernelConv2D_3/bias
BN_3/gamma	BN_3/betaBN_3/moving_meanBN_3/moving_varianceConv2D_4/kernelConv2D_4/bias
BN_4/gamma	BN_4/betaBN_4/moving_meanBN_4/moving_varianceConv2D_5/kernelConv2D_5/bias
BN_5/gamma	BN_5/betaBN_5/moving_meanBN_5/moving_varianceDense_Layer_6/kernelDense_Layer_6/biasDense_MU/kernelDense_MU/biasDense_Log_Var/kernelDense_Log_Var/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:?????????:?????????:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_401186
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
α
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#Conv2D_1/kernel/Read/ReadVariableOp!Conv2D_1/bias/Read/ReadVariableOpBN_1/gamma/Read/ReadVariableOpBN_1/beta/Read/ReadVariableOp$BN_1/moving_mean/Read/ReadVariableOp(BN_1/moving_variance/Read/ReadVariableOp#Conv2D_2/kernel/Read/ReadVariableOp!Conv2D_2/bias/Read/ReadVariableOpBN_2/gamma/Read/ReadVariableOpBN_2/beta/Read/ReadVariableOp$BN_2/moving_mean/Read/ReadVariableOp(BN_2/moving_variance/Read/ReadVariableOp#Conv2D_3/kernel/Read/ReadVariableOp!Conv2D_3/bias/Read/ReadVariableOpBN_3/gamma/Read/ReadVariableOpBN_3/beta/Read/ReadVariableOp$BN_3/moving_mean/Read/ReadVariableOp(BN_3/moving_variance/Read/ReadVariableOp#Conv2D_4/kernel/Read/ReadVariableOp!Conv2D_4/bias/Read/ReadVariableOpBN_4/gamma/Read/ReadVariableOpBN_4/beta/Read/ReadVariableOp$BN_4/moving_mean/Read/ReadVariableOp(BN_4/moving_variance/Read/ReadVariableOp#Conv2D_5/kernel/Read/ReadVariableOp!Conv2D_5/bias/Read/ReadVariableOpBN_5/gamma/Read/ReadVariableOpBN_5/beta/Read/ReadVariableOp$BN_5/moving_mean/Read/ReadVariableOp(BN_5/moving_variance/Read/ReadVariableOp(Dense_Layer_6/kernel/Read/ReadVariableOp&Dense_Layer_6/bias/Read/ReadVariableOp#Dense_MU/kernel/Read/ReadVariableOp!Dense_MU/bias/Read/ReadVariableOp(Dense_Log_Var/kernel/Read/ReadVariableOp&Dense_Log_Var/bias/Read/ReadVariableOpConst*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_402290

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv2D_1/kernelConv2D_1/bias
BN_1/gamma	BN_1/betaBN_1/moving_meanBN_1/moving_varianceConv2D_2/kernelConv2D_2/bias
BN_2/gamma	BN_2/betaBN_2/moving_meanBN_2/moving_varianceConv2D_3/kernelConv2D_3/bias
BN_3/gamma	BN_3/betaBN_3/moving_meanBN_3/moving_varianceConv2D_4/kernelConv2D_4/bias
BN_4/gamma	BN_4/betaBN_4/moving_meanBN_4/moving_varianceConv2D_5/kernelConv2D_5/bias
BN_5/gamma	BN_5/betaBN_5/moving_meanBN_5/moving_varianceDense_Layer_6/kernelDense_Layer_6/biasDense_MU/kernelDense_MU/biasDense_Log_Var/kernelDense_Log_Var/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_402408ν
τ
‘
)__inference_Conv2D_4_layer_call_fn_401889

inputs#
unknown:
	unknown_0:	
identity’StatefulPartitionedCallε
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_4_layer_call_and_return_conditional_losses_400268x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
P
κ
C__inference_encoder_layer_call_and_return_conditional_losses_401010
encoder_input)
conv2d_1_400920: 
conv2d_1_400922: 
bn_1_400925: 
bn_1_400927: 
bn_1_400929: 
bn_1_400931: )
conv2d_2_400934: @
conv2d_2_400936:@
bn_2_400939:@
bn_2_400941:@
bn_2_400943:@
bn_2_400945:@*
conv2d_3_400948:@
conv2d_3_400950:	
bn_3_400953:	
bn_3_400955:	
bn_3_400957:	
bn_3_400959:	+
conv2d_4_400962:
conv2d_4_400964:	
bn_4_400967:	
bn_4_400969:	
bn_4_400971:	
bn_4_400973:	+
conv2d_5_400976:
conv2d_5_400978:	
bn_5_400981:	
bn_5_400983:	
bn_5_400985:	
bn_5_400987:	(
dense_layer_6_400991:
@#
dense_layer_6_400993:	#
dense_mu_400996:

dense_mu_400998:	(
dense_log_var_401001:
#
dense_log_var_401003:	
identity

identity_1

identity_2’BN_1/StatefulPartitionedCall’BN_2/StatefulPartitionedCall’BN_3/StatefulPartitionedCall’BN_4/StatefulPartitionedCall’BN_5/StatefulPartitionedCall’Code/StatefulPartitionedCall’ Conv2D_1/StatefulPartitionedCall’ Conv2D_2/StatefulPartitionedCall’ Conv2D_3/StatefulPartitionedCall’ Conv2D_4/StatefulPartitionedCall’ Conv2D_5/StatefulPartitionedCall’%Dense_Layer_6/StatefulPartitionedCall’%Dense_Log_Var/StatefulPartitionedCall’ Dense_MU/StatefulPartitionedCall
 Conv2D_1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputconv2d_1_400920conv2d_1_400922*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_400190¬
BN_1/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_1/StatefulPartitionedCall:output:0bn_1_400925bn_1_400927bn_1_400929bn_1_400931*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_1_layer_call_and_return_conditional_losses_399874
 Conv2D_2/StatefulPartitionedCallStatefulPartitionedCall%BN_1/StatefulPartitionedCall:output:0conv2d_2_400934conv2d_2_400936*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_2_layer_call_and_return_conditional_losses_400216¬
BN_2/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_2/StatefulPartitionedCall:output:0bn_2_400939bn_2_400941bn_2_400943bn_2_400945*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_2_layer_call_and_return_conditional_losses_399938
 Conv2D_3/StatefulPartitionedCallStatefulPartitionedCall%BN_2/StatefulPartitionedCall:output:0conv2d_3_400948conv2d_3_400950*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_3_layer_call_and_return_conditional_losses_400242­
BN_3/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_3/StatefulPartitionedCall:output:0bn_3_400953bn_3_400955bn_3_400957bn_3_400959*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_3_layer_call_and_return_conditional_losses_400002
 Conv2D_4/StatefulPartitionedCallStatefulPartitionedCall%BN_3/StatefulPartitionedCall:output:0conv2d_4_400962conv2d_4_400964*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_4_layer_call_and_return_conditional_losses_400268­
BN_4/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_4/StatefulPartitionedCall:output:0bn_4_400967bn_4_400969bn_4_400971bn_4_400973*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_4_layer_call_and_return_conditional_losses_400066
 Conv2D_5/StatefulPartitionedCallStatefulPartitionedCall%BN_4/StatefulPartitionedCall:output:0conv2d_5_400976conv2d_5_400978*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_5_layer_call_and_return_conditional_losses_400294­
BN_5/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_5/StatefulPartitionedCall:output:0bn_5_400981bn_5_400983bn_5_400985bn_5_400987*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_5_layer_call_and_return_conditional_losses_400130Ω
Flatten/PartitionedCallPartitionedCall%BN_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Flatten_layer_call_and_return_conditional_losses_400315’
%Dense_Layer_6/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0dense_layer_6_400991dense_layer_6_400993*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_Dense_Layer_6_layer_call_and_return_conditional_losses_400328
 Dense_MU/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_6/StatefulPartitionedCall:output:0dense_mu_400996dense_mu_400998*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Dense_MU_layer_call_and_return_conditional_losses_400344°
%Dense_Log_Var/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_6/StatefulPartitionedCall:output:0dense_log_var_401001dense_log_var_401003*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_Dense_Log_Var_layer_call_and_return_conditional_losses_400360
Code/StatefulPartitionedCallStatefulPartitionedCall)Dense_MU/StatefulPartitionedCall:output:0.Dense_Log_Var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Code_layer_call_and_return_conditional_losses_400382y
IdentityIdentity)Dense_MU/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????

Identity_1Identity.Dense_Log_Var/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????w

Identity_2Identity%Code/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????’
NoOpNoOp^BN_1/StatefulPartitionedCall^BN_2/StatefulPartitionedCall^BN_3/StatefulPartitionedCall^BN_4/StatefulPartitionedCall^BN_5/StatefulPartitionedCall^Code/StatefulPartitionedCall!^Conv2D_1/StatefulPartitionedCall!^Conv2D_2/StatefulPartitionedCall!^Conv2D_3/StatefulPartitionedCall!^Conv2D_4/StatefulPartitionedCall!^Conv2D_5/StatefulPartitionedCall&^Dense_Layer_6/StatefulPartitionedCall&^Dense_Log_Var/StatefulPartitionedCall!^Dense_MU/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
BN_1/StatefulPartitionedCallBN_1/StatefulPartitionedCall2<
BN_2/StatefulPartitionedCallBN_2/StatefulPartitionedCall2<
BN_3/StatefulPartitionedCallBN_3/StatefulPartitionedCall2<
BN_4/StatefulPartitionedCallBN_4/StatefulPartitionedCall2<
BN_5/StatefulPartitionedCallBN_5/StatefulPartitionedCall2<
Code/StatefulPartitionedCallCode/StatefulPartitionedCall2D
 Conv2D_1/StatefulPartitionedCall Conv2D_1/StatefulPartitionedCall2D
 Conv2D_2/StatefulPartitionedCall Conv2D_2/StatefulPartitionedCall2D
 Conv2D_3/StatefulPartitionedCall Conv2D_3/StatefulPartitionedCall2D
 Conv2D_4/StatefulPartitionedCall Conv2D_4/StatefulPartitionedCall2D
 Conv2D_5/StatefulPartitionedCall Conv2D_5/StatefulPartitionedCall2N
%Dense_Layer_6/StatefulPartitionedCall%Dense_Layer_6/StatefulPartitionedCall2N
%Dense_Log_Var/StatefulPartitionedCall%Dense_Log_Var/StatefulPartitionedCall2D
 Dense_MU/StatefulPartitionedCall Dense_MU/StatefulPartitionedCall:` \
1
_output_shapes
:?????????
'
_user_specified_nameEncoder_Input

³
@__inference_BN_3_layer_call_and_return_conditional_losses_401880

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’AssignNewValue’AssignNewValue_1’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ϋ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<Ζ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Π
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????Τ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ψ
Δ
%__inference_BN_3_layer_call_fn_401844

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_3_layer_call_and_return_conditional_losses_400033
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs

³
@__inference_BN_3_layer_call_and_return_conditional_losses_400033

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’AssignNewValue’AssignNewValue_1’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ϋ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<Ζ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Π
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????Τ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ͺ
o
@__inference_Code_layer_call_and_return_conditional_losses_402141
inputs_0
inputs_1
identity=
ShapeShapeinputs_0*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*(
_output_shapes
:?????????*
dtype0
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*(
_output_shapes
:?????????}
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*(
_output_shapes
:?????????J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?W
mulMulinputs_1mul/y:output:0*
T0*(
_output_shapes
:?????????F
ExpExpmul:z:0*
T0*(
_output_shapes
:?????????[
mul_1MulExp:y:0random_normal:z:0*
T0*(
_output_shapes
:?????????T
addAddV2inputs_0	mul_1:z:0*
T0*(
_output_shapes
:?????????P
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????:?????????:R N
(
_output_shapes
:?????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:?????????
"
_user_specified_name
inputs/1
Λ

@__inference_BN_3_layer_call_and_return_conditional_losses_400002

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ν
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Ώ
	
(__inference_encoder_layer_call_fn_400917
encoder_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:
@

unknown_30:	

unknown_31:


unknown_32:	

unknown_33:


unknown_34:	
identity

identity_1

identity_2’StatefulPartitionedCallΦ
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:?????????:?????????:?????????*<
_read_only_resource_inputs
	
 !"#$*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_400757p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:?????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
1
_output_shapes
:?????????
'
_user_specified_nameEncoder_Input
Λ

@__inference_BN_4_layer_call_and_return_conditional_losses_400066

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ν
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ϊ
Δ
%__inference_BN_4_layer_call_fn_401913

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_4_layer_call_and_return_conditional_losses_400066
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
υ
n
%__inference_Code_layer_call_fn_402119
inputs_0
inputs_1
identity’StatefulPartitionedCallΜ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Code_layer_call_and_return_conditional_losses_400382p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????:?????????22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:?????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:?????????
"
_user_specified_name
inputs/1
΄
	
(__inference_encoder_layer_call_fn_401267

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:
@

unknown_30:	

unknown_31:


unknown_32:	

unknown_33:


unknown_34:	
identity

identity_1

identity_2’StatefulPartitionedCallΩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:?????????:?????????:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_400387p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:?????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs
ψ
Δ
%__inference_BN_5_layer_call_fn_402008

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_5_layer_call_and_return_conditional_losses_400161
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
υ
n
%__inference_Code_layer_call_fn_402125
inputs_0
inputs_1
identity’StatefulPartitionedCallΜ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Code_layer_call_and_return_conditional_losses_400492p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????:?????????22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:?????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:?????????
"
_user_specified_name
inputs/1
τ
‘
)__inference_Conv2D_5_layer_call_fn_401971

inputs#
unknown:
	unknown_0:	
identity’StatefulPartitionedCallε
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_5_layer_call_and_return_conditional_losses_400294x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Φ

.__inference_Dense_Log_Var_layer_call_fn_402103

inputs
unknown:

	unknown_0:	
identity’StatefulPartitionedCallβ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_Dense_Log_Var_layer_call_and_return_conditional_losses_400360p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ϊ
Δ
%__inference_BN_3_layer_call_fn_401831

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_3_layer_call_and_return_conditional_losses_400002
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs


D__inference_Conv2D_4_layer_call_and_return_conditional_losses_400268

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


D__inference_Conv2D_5_layer_call_and_return_conditional_losses_401982

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Λ

@__inference_BN_5_layer_call_and_return_conditional_losses_402026

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ν
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
»

@__inference_BN_1_layer_call_and_return_conditional_losses_401698

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Θ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs


D__inference_Conv2D_4_layer_call_and_return_conditional_losses_401900

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ς
Γ
C__inference_encoder_layer_call_and_return_conditional_losses_401491

inputsA
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: *
bn_1_readvariableop_resource: ,
bn_1_readvariableop_1_resource: ;
-bn_1_fusedbatchnormv3_readvariableop_resource: =
/bn_1_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@*
bn_2_readvariableop_resource:@,
bn_2_readvariableop_1_resource:@;
-bn_2_fusedbatchnormv3_readvariableop_resource:@=
/bn_2_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_3_conv2d_readvariableop_resource:@7
(conv2d_3_biasadd_readvariableop_resource:	+
bn_3_readvariableop_resource:	-
bn_3_readvariableop_1_resource:	<
-bn_3_fusedbatchnormv3_readvariableop_resource:	>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_4_conv2d_readvariableop_resource:7
(conv2d_4_biasadd_readvariableop_resource:	+
bn_4_readvariableop_resource:	-
bn_4_readvariableop_1_resource:	<
-bn_4_fusedbatchnormv3_readvariableop_resource:	>
/bn_4_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_5_conv2d_readvariableop_resource:7
(conv2d_5_biasadd_readvariableop_resource:	+
bn_5_readvariableop_resource:	-
bn_5_readvariableop_1_resource:	<
-bn_5_fusedbatchnormv3_readvariableop_resource:	>
/bn_5_fusedbatchnormv3_readvariableop_1_resource:	@
,dense_layer_6_matmul_readvariableop_resource:
@<
-dense_layer_6_biasadd_readvariableop_resource:	;
'dense_mu_matmul_readvariableop_resource:
7
(dense_mu_biasadd_readvariableop_resource:	@
,dense_log_var_matmul_readvariableop_resource:
<
-dense_log_var_biasadd_readvariableop_resource:	
identity

identity_1

identity_2’$BN_1/FusedBatchNormV3/ReadVariableOp’&BN_1/FusedBatchNormV3/ReadVariableOp_1’BN_1/ReadVariableOp’BN_1/ReadVariableOp_1’$BN_2/FusedBatchNormV3/ReadVariableOp’&BN_2/FusedBatchNormV3/ReadVariableOp_1’BN_2/ReadVariableOp’BN_2/ReadVariableOp_1’$BN_3/FusedBatchNormV3/ReadVariableOp’&BN_3/FusedBatchNormV3/ReadVariableOp_1’BN_3/ReadVariableOp’BN_3/ReadVariableOp_1’$BN_4/FusedBatchNormV3/ReadVariableOp’&BN_4/FusedBatchNormV3/ReadVariableOp_1’BN_4/ReadVariableOp’BN_4/ReadVariableOp_1’$BN_5/FusedBatchNormV3/ReadVariableOp’&BN_5/FusedBatchNormV3/ReadVariableOp_1’BN_5/ReadVariableOp’BN_5/ReadVariableOp_1’Conv2D_1/BiasAdd/ReadVariableOp’Conv2D_1/Conv2D/ReadVariableOp’Conv2D_2/BiasAdd/ReadVariableOp’Conv2D_2/Conv2D/ReadVariableOp’Conv2D_3/BiasAdd/ReadVariableOp’Conv2D_3/Conv2D/ReadVariableOp’Conv2D_4/BiasAdd/ReadVariableOp’Conv2D_4/Conv2D/ReadVariableOp’Conv2D_5/BiasAdd/ReadVariableOp’Conv2D_5/Conv2D/ReadVariableOp’$Dense_Layer_6/BiasAdd/ReadVariableOp’#Dense_Layer_6/MatMul/ReadVariableOp’$Dense_Log_Var/BiasAdd/ReadVariableOp’#Dense_Log_Var/MatMul/ReadVariableOp’Dense_MU/BiasAdd/ReadVariableOp’Dense_MU/MatMul/ReadVariableOp
Conv2D_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0«
Conv2D_1/Conv2DConv2Dinputs&Conv2D_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides

Conv2D_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
Conv2D_1/BiasAddBiasAddConv2D_1/Conv2D:output:0'Conv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ j
Conv2D_1/ReluReluConv2D_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ l
BN_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes
: *
dtype0p
BN_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes
: *
dtype0
$BN_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
&BN_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0δ
BN_1/FusedBatchNormV3FusedBatchNormV3Conv2D_1/Relu:activations:0BN_1/ReadVariableOp:value:0BN_1/ReadVariableOp_1:value:0,BN_1/FusedBatchNormV3/ReadVariableOp:value:0.BN_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o:*
is_training( 
Conv2D_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ύ
Conv2D_2/Conv2DConv2DBN_1/FusedBatchNormV3:y:0&Conv2D_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides

Conv2D_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
Conv2D_2/BiasAddBiasAddConv2D_2/Conv2D:output:0'Conv2D_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @j
Conv2D_2/ReluReluConv2D_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @l
BN_2/ReadVariableOpReadVariableOpbn_2_readvariableop_resource*
_output_shapes
:@*
dtype0p
BN_2/ReadVariableOp_1ReadVariableOpbn_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0
$BN_2/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
&BN_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0δ
BN_2/FusedBatchNormV3FusedBatchNormV3Conv2D_2/Relu:activations:0BN_2/ReadVariableOp:value:0BN_2/ReadVariableOp_1:value:0,BN_2/FusedBatchNormV3/ReadVariableOp:value:0.BN_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o:*
is_training( 
Conv2D_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ώ
Conv2D_3/Conv2DConv2DBN_2/FusedBatchNormV3:y:0&Conv2D_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

Conv2D_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
Conv2D_3/BiasAddBiasAddConv2D_3/Conv2D:output:0'Conv2D_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????k
Conv2D_3/ReluReluConv2D_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????m
BN_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:*
dtype0q
BN_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
$BN_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
&BN_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ι
BN_3/FusedBatchNormV3FusedBatchNormV3Conv2D_3/Relu:activations:0BN_3/ReadVariableOp:value:0BN_3/ReadVariableOp_1:value:0,BN_3/FusedBatchNormV3/ReadVariableOp:value:0.BN_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 
Conv2D_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ώ
Conv2D_4/Conv2DConv2DBN_3/FusedBatchNormV3:y:0&Conv2D_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

Conv2D_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
Conv2D_4/BiasAddBiasAddConv2D_4/Conv2D:output:0'Conv2D_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????k
Conv2D_4/ReluReluConv2D_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????m
BN_4/ReadVariableOpReadVariableOpbn_4_readvariableop_resource*
_output_shapes	
:*
dtype0q
BN_4/ReadVariableOp_1ReadVariableOpbn_4_readvariableop_1_resource*
_output_shapes	
:*
dtype0
$BN_4/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
&BN_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ι
BN_4/FusedBatchNormV3FusedBatchNormV3Conv2D_4/Relu:activations:0BN_4/ReadVariableOp:value:0BN_4/ReadVariableOp_1:value:0,BN_4/FusedBatchNormV3/ReadVariableOp:value:0.BN_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 
Conv2D_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ώ
Conv2D_5/Conv2DConv2DBN_4/FusedBatchNormV3:y:0&Conv2D_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

Conv2D_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
Conv2D_5/BiasAddBiasAddConv2D_5/Conv2D:output:0'Conv2D_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????k
Conv2D_5/ReluReluConv2D_5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????m
BN_5/ReadVariableOpReadVariableOpbn_5_readvariableop_resource*
_output_shapes	
:*
dtype0q
BN_5/ReadVariableOp_1ReadVariableOpbn_5_readvariableop_1_resource*
_output_shapes	
:*
dtype0
$BN_5/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
&BN_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ι
BN_5/FusedBatchNormV3FusedBatchNormV3Conv2D_5/Relu:activations:0BN_5/ReadVariableOp:value:0BN_5/ReadVariableOp_1:value:0,BN_5/FusedBatchNormV3/ReadVariableOp:value:0.BN_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( ^
Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    
Flatten/ReshapeReshapeBN_5/FusedBatchNormV3:y:0Flatten/Const:output:0*
T0*(
_output_shapes
:?????????@
#Dense_Layer_6/MatMul/ReadVariableOpReadVariableOp,dense_layer_6_matmul_readvariableop_resource* 
_output_shapes
:
@*
dtype0
Dense_Layer_6/MatMulMatMulFlatten/Reshape:output:0+Dense_Layer_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$Dense_Layer_6/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0‘
Dense_Layer_6/BiasAddBiasAddDense_Layer_6/MatMul:product:0,Dense_Layer_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????m
Dense_Layer_6/ReluReluDense_Layer_6/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
Dense_MU/MatMul/ReadVariableOpReadVariableOp'dense_mu_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
Dense_MU/MatMulMatMul Dense_Layer_6/Relu:activations:0&Dense_MU/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
Dense_MU/BiasAdd/ReadVariableOpReadVariableOp(dense_mu_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
Dense_MU/BiasAddBiasAddDense_MU/MatMul:product:0'Dense_MU/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
#Dense_Log_Var/MatMul/ReadVariableOpReadVariableOp,dense_log_var_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0 
Dense_Log_Var/MatMulMatMul Dense_Layer_6/Relu:activations:0+Dense_Log_Var/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$Dense_Log_Var/BiasAdd/ReadVariableOpReadVariableOp-dense_log_var_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0‘
Dense_Log_Var/BiasAddBiasAddDense_Log_Var/MatMul:product:0,Dense_Log_Var/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????S

Code/ShapeShapeDense_MU/BiasAdd:output:0*
T0*
_output_shapes
:\
Code/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
Code/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
'Code/random_normal/RandomStandardNormalRandomStandardNormalCode/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0¦
Code/random_normal/mulMul0Code/random_normal/RandomStandardNormal:output:0"Code/random_normal/stddev:output:0*
T0*(
_output_shapes
:?????????
Code/random_normalAddV2Code/random_normal/mul:z:0 Code/random_normal/mean:output:0*
T0*(
_output_shapes
:?????????O

Code/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
Code/mulMulDense_Log_Var/BiasAdd:output:0Code/mul/y:output:0*
T0*(
_output_shapes
:?????????P
Code/ExpExpCode/mul:z:0*
T0*(
_output_shapes
:?????????j

Code/mul_1MulCode/Exp:y:0Code/random_normal:z:0*
T0*(
_output_shapes
:?????????o
Code/addAddV2Dense_MU/BiasAdd:output:0Code/mul_1:z:0*
T0*(
_output_shapes
:?????????i
IdentityIdentityDense_MU/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:?????????p

Identity_1IdentityDense_Log_Var/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:?????????^

Identity_2IdentityCode/add:z:0^NoOp*
T0*(
_output_shapes
:?????????θ	
NoOpNoOp%^BN_1/FusedBatchNormV3/ReadVariableOp'^BN_1/FusedBatchNormV3/ReadVariableOp_1^BN_1/ReadVariableOp^BN_1/ReadVariableOp_1%^BN_2/FusedBatchNormV3/ReadVariableOp'^BN_2/FusedBatchNormV3/ReadVariableOp_1^BN_2/ReadVariableOp^BN_2/ReadVariableOp_1%^BN_3/FusedBatchNormV3/ReadVariableOp'^BN_3/FusedBatchNormV3/ReadVariableOp_1^BN_3/ReadVariableOp^BN_3/ReadVariableOp_1%^BN_4/FusedBatchNormV3/ReadVariableOp'^BN_4/FusedBatchNormV3/ReadVariableOp_1^BN_4/ReadVariableOp^BN_4/ReadVariableOp_1%^BN_5/FusedBatchNormV3/ReadVariableOp'^BN_5/FusedBatchNormV3/ReadVariableOp_1^BN_5/ReadVariableOp^BN_5/ReadVariableOp_1 ^Conv2D_1/BiasAdd/ReadVariableOp^Conv2D_1/Conv2D/ReadVariableOp ^Conv2D_2/BiasAdd/ReadVariableOp^Conv2D_2/Conv2D/ReadVariableOp ^Conv2D_3/BiasAdd/ReadVariableOp^Conv2D_3/Conv2D/ReadVariableOp ^Conv2D_4/BiasAdd/ReadVariableOp^Conv2D_4/Conv2D/ReadVariableOp ^Conv2D_5/BiasAdd/ReadVariableOp^Conv2D_5/Conv2D/ReadVariableOp%^Dense_Layer_6/BiasAdd/ReadVariableOp$^Dense_Layer_6/MatMul/ReadVariableOp%^Dense_Log_Var/BiasAdd/ReadVariableOp$^Dense_Log_Var/MatMul/ReadVariableOp ^Dense_MU/BiasAdd/ReadVariableOp^Dense_MU/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$BN_1/FusedBatchNormV3/ReadVariableOp$BN_1/FusedBatchNormV3/ReadVariableOp2P
&BN_1/FusedBatchNormV3/ReadVariableOp_1&BN_1/FusedBatchNormV3/ReadVariableOp_12*
BN_1/ReadVariableOpBN_1/ReadVariableOp2.
BN_1/ReadVariableOp_1BN_1/ReadVariableOp_12L
$BN_2/FusedBatchNormV3/ReadVariableOp$BN_2/FusedBatchNormV3/ReadVariableOp2P
&BN_2/FusedBatchNormV3/ReadVariableOp_1&BN_2/FusedBatchNormV3/ReadVariableOp_12*
BN_2/ReadVariableOpBN_2/ReadVariableOp2.
BN_2/ReadVariableOp_1BN_2/ReadVariableOp_12L
$BN_3/FusedBatchNormV3/ReadVariableOp$BN_3/FusedBatchNormV3/ReadVariableOp2P
&BN_3/FusedBatchNormV3/ReadVariableOp_1&BN_3/FusedBatchNormV3/ReadVariableOp_12*
BN_3/ReadVariableOpBN_3/ReadVariableOp2.
BN_3/ReadVariableOp_1BN_3/ReadVariableOp_12L
$BN_4/FusedBatchNormV3/ReadVariableOp$BN_4/FusedBatchNormV3/ReadVariableOp2P
&BN_4/FusedBatchNormV3/ReadVariableOp_1&BN_4/FusedBatchNormV3/ReadVariableOp_12*
BN_4/ReadVariableOpBN_4/ReadVariableOp2.
BN_4/ReadVariableOp_1BN_4/ReadVariableOp_12L
$BN_5/FusedBatchNormV3/ReadVariableOp$BN_5/FusedBatchNormV3/ReadVariableOp2P
&BN_5/FusedBatchNormV3/ReadVariableOp_1&BN_5/FusedBatchNormV3/ReadVariableOp_12*
BN_5/ReadVariableOpBN_5/ReadVariableOp2.
BN_5/ReadVariableOp_1BN_5/ReadVariableOp_12B
Conv2D_1/BiasAdd/ReadVariableOpConv2D_1/BiasAdd/ReadVariableOp2@
Conv2D_1/Conv2D/ReadVariableOpConv2D_1/Conv2D/ReadVariableOp2B
Conv2D_2/BiasAdd/ReadVariableOpConv2D_2/BiasAdd/ReadVariableOp2@
Conv2D_2/Conv2D/ReadVariableOpConv2D_2/Conv2D/ReadVariableOp2B
Conv2D_3/BiasAdd/ReadVariableOpConv2D_3/BiasAdd/ReadVariableOp2@
Conv2D_3/Conv2D/ReadVariableOpConv2D_3/Conv2D/ReadVariableOp2B
Conv2D_4/BiasAdd/ReadVariableOpConv2D_4/BiasAdd/ReadVariableOp2@
Conv2D_4/Conv2D/ReadVariableOpConv2D_4/Conv2D/ReadVariableOp2B
Conv2D_5/BiasAdd/ReadVariableOpConv2D_5/BiasAdd/ReadVariableOp2@
Conv2D_5/Conv2D/ReadVariableOpConv2D_5/Conv2D/ReadVariableOp2L
$Dense_Layer_6/BiasAdd/ReadVariableOp$Dense_Layer_6/BiasAdd/ReadVariableOp2J
#Dense_Layer_6/MatMul/ReadVariableOp#Dense_Layer_6/MatMul/ReadVariableOp2L
$Dense_Log_Var/BiasAdd/ReadVariableOp$Dense_Log_Var/BiasAdd/ReadVariableOp2J
#Dense_Log_Var/MatMul/ReadVariableOp#Dense_Log_Var/MatMul/ReadVariableOp2B
Dense_MU/BiasAdd/ReadVariableOpDense_MU/BiasAdd/ReadVariableOp2@
Dense_MU/MatMul/ReadVariableOpDense_MU/MatMul/ReadVariableOp:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs
Ύ
Ε
"__inference__traced_restore_402408
file_prefix:
 assignvariableop_conv2d_1_kernel: .
 assignvariableop_1_conv2d_1_bias: +
assignvariableop_2_bn_1_gamma: *
assignvariableop_3_bn_1_beta: 1
#assignvariableop_4_bn_1_moving_mean: 5
'assignvariableop_5_bn_1_moving_variance: <
"assignvariableop_6_conv2d_2_kernel: @.
 assignvariableop_7_conv2d_2_bias:@+
assignvariableop_8_bn_2_gamma:@*
assignvariableop_9_bn_2_beta:@2
$assignvariableop_10_bn_2_moving_mean:@6
(assignvariableop_11_bn_2_moving_variance:@>
#assignvariableop_12_conv2d_3_kernel:@0
!assignvariableop_13_conv2d_3_bias:	-
assignvariableop_14_bn_3_gamma:	,
assignvariableop_15_bn_3_beta:	3
$assignvariableop_16_bn_3_moving_mean:	7
(assignvariableop_17_bn_3_moving_variance:	?
#assignvariableop_18_conv2d_4_kernel:0
!assignvariableop_19_conv2d_4_bias:	-
assignvariableop_20_bn_4_gamma:	,
assignvariableop_21_bn_4_beta:	3
$assignvariableop_22_bn_4_moving_mean:	7
(assignvariableop_23_bn_4_moving_variance:	?
#assignvariableop_24_conv2d_5_kernel:0
!assignvariableop_25_conv2d_5_bias:	-
assignvariableop_26_bn_5_gamma:	,
assignvariableop_27_bn_5_beta:	3
$assignvariableop_28_bn_5_moving_mean:	7
(assignvariableop_29_bn_5_moving_variance:	<
(assignvariableop_30_dense_layer_6_kernel:
@5
&assignvariableop_31_dense_layer_6_bias:	7
#assignvariableop_32_dense_mu_kernel:
0
!assignvariableop_33_dense_mu_bias:	<
(assignvariableop_34_dense_log_var_kernel:
5
&assignvariableop_35_dense_log_var_bias:	
identity_37’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*ΐ
valueΆB³%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHΊ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ϊ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ͺ
_output_shapes
:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_conv2d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_bn_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_bn_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp#assignvariableop_4_bn_1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp'assignvariableop_5_bn_1_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_bn_2_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_bn_2_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp$assignvariableop_10_bn_2_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp(assignvariableop_11_bn_2_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_bn_3_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_bn_3_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp$assignvariableop_16_bn_3_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp(assignvariableop_17_bn_3_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2d_4_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2d_4_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_bn_4_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_bn_4_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp$assignvariableop_22_bn_4_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp(assignvariableop_23_bn_4_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv2d_5_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv2d_5_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_bn_5_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_bn_5_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp$assignvariableop_28_bn_5_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp(assignvariableop_29_bn_5_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_dense_layer_6_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp&assignvariableop_31_dense_layer_6_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp#assignvariableop_32_dense_mu_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp!assignvariableop_33_dense_mu_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp(assignvariableop_34_dense_log_var_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp&assignvariableop_35_dense_log_var_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 η
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: Τ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_37Identity_37:output:0*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

ύ
D__inference_Conv2D_2_layer_call_and_return_conditional_losses_401736

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????  @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs

?
D__inference_Conv2D_3_layer_call_and_return_conditional_losses_401818

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs

³
@__inference_BN_4_layer_call_and_return_conditional_losses_400097

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’AssignNewValue’AssignNewValue_1’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ϋ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<Ζ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Π
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????Τ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
εO
γ
C__inference_encoder_layer_call_and_return_conditional_losses_400757

inputs)
conv2d_1_400667: 
conv2d_1_400669: 
bn_1_400672: 
bn_1_400674: 
bn_1_400676: 
bn_1_400678: )
conv2d_2_400681: @
conv2d_2_400683:@
bn_2_400686:@
bn_2_400688:@
bn_2_400690:@
bn_2_400692:@*
conv2d_3_400695:@
conv2d_3_400697:	
bn_3_400700:	
bn_3_400702:	
bn_3_400704:	
bn_3_400706:	+
conv2d_4_400709:
conv2d_4_400711:	
bn_4_400714:	
bn_4_400716:	
bn_4_400718:	
bn_4_400720:	+
conv2d_5_400723:
conv2d_5_400725:	
bn_5_400728:	
bn_5_400730:	
bn_5_400732:	
bn_5_400734:	(
dense_layer_6_400738:
@#
dense_layer_6_400740:	#
dense_mu_400743:

dense_mu_400745:	(
dense_log_var_400748:
#
dense_log_var_400750:	
identity

identity_1

identity_2’BN_1/StatefulPartitionedCall’BN_2/StatefulPartitionedCall’BN_3/StatefulPartitionedCall’BN_4/StatefulPartitionedCall’BN_5/StatefulPartitionedCall’Code/StatefulPartitionedCall’ Conv2D_1/StatefulPartitionedCall’ Conv2D_2/StatefulPartitionedCall’ Conv2D_3/StatefulPartitionedCall’ Conv2D_4/StatefulPartitionedCall’ Conv2D_5/StatefulPartitionedCall’%Dense_Layer_6/StatefulPartitionedCall’%Dense_Log_Var/StatefulPartitionedCall’ Dense_MU/StatefulPartitionedCallϋ
 Conv2D_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_400667conv2d_1_400669*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_400190ͺ
BN_1/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_1/StatefulPartitionedCall:output:0bn_1_400672bn_1_400674bn_1_400676bn_1_400678*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_1_layer_call_and_return_conditional_losses_399905
 Conv2D_2/StatefulPartitionedCallStatefulPartitionedCall%BN_1/StatefulPartitionedCall:output:0conv2d_2_400681conv2d_2_400683*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_2_layer_call_and_return_conditional_losses_400216ͺ
BN_2/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_2/StatefulPartitionedCall:output:0bn_2_400686bn_2_400688bn_2_400690bn_2_400692*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_2_layer_call_and_return_conditional_losses_399969
 Conv2D_3/StatefulPartitionedCallStatefulPartitionedCall%BN_2/StatefulPartitionedCall:output:0conv2d_3_400695conv2d_3_400697*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_3_layer_call_and_return_conditional_losses_400242«
BN_3/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_3/StatefulPartitionedCall:output:0bn_3_400700bn_3_400702bn_3_400704bn_3_400706*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_3_layer_call_and_return_conditional_losses_400033
 Conv2D_4/StatefulPartitionedCallStatefulPartitionedCall%BN_3/StatefulPartitionedCall:output:0conv2d_4_400709conv2d_4_400711*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_4_layer_call_and_return_conditional_losses_400268«
BN_4/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_4/StatefulPartitionedCall:output:0bn_4_400714bn_4_400716bn_4_400718bn_4_400720*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_4_layer_call_and_return_conditional_losses_400097
 Conv2D_5/StatefulPartitionedCallStatefulPartitionedCall%BN_4/StatefulPartitionedCall:output:0conv2d_5_400723conv2d_5_400725*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_5_layer_call_and_return_conditional_losses_400294«
BN_5/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_5/StatefulPartitionedCall:output:0bn_5_400728bn_5_400730bn_5_400732bn_5_400734*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_5_layer_call_and_return_conditional_losses_400161Ω
Flatten/PartitionedCallPartitionedCall%BN_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Flatten_layer_call_and_return_conditional_losses_400315’
%Dense_Layer_6/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0dense_layer_6_400738dense_layer_6_400740*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_Dense_Layer_6_layer_call_and_return_conditional_losses_400328
 Dense_MU/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_6/StatefulPartitionedCall:output:0dense_mu_400743dense_mu_400745*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Dense_MU_layer_call_and_return_conditional_losses_400344°
%Dense_Log_Var/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_6/StatefulPartitionedCall:output:0dense_log_var_400748dense_log_var_400750*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_Dense_Log_Var_layer_call_and_return_conditional_losses_400360
Code/StatefulPartitionedCallStatefulPartitionedCall)Dense_MU/StatefulPartitionedCall:output:0.Dense_Log_Var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Code_layer_call_and_return_conditional_losses_400492y
IdentityIdentity)Dense_MU/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????

Identity_1Identity.Dense_Log_Var/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????w

Identity_2Identity%Code/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????’
NoOpNoOp^BN_1/StatefulPartitionedCall^BN_2/StatefulPartitionedCall^BN_3/StatefulPartitionedCall^BN_4/StatefulPartitionedCall^BN_5/StatefulPartitionedCall^Code/StatefulPartitionedCall!^Conv2D_1/StatefulPartitionedCall!^Conv2D_2/StatefulPartitionedCall!^Conv2D_3/StatefulPartitionedCall!^Conv2D_4/StatefulPartitionedCall!^Conv2D_5/StatefulPartitionedCall&^Dense_Layer_6/StatefulPartitionedCall&^Dense_Log_Var/StatefulPartitionedCall!^Dense_MU/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
BN_1/StatefulPartitionedCallBN_1/StatefulPartitionedCall2<
BN_2/StatefulPartitionedCallBN_2/StatefulPartitionedCall2<
BN_3/StatefulPartitionedCallBN_3/StatefulPartitionedCall2<
BN_4/StatefulPartitionedCallBN_4/StatefulPartitionedCall2<
BN_5/StatefulPartitionedCallBN_5/StatefulPartitionedCall2<
Code/StatefulPartitionedCallCode/StatefulPartitionedCall2D
 Conv2D_1/StatefulPartitionedCall Conv2D_1/StatefulPartitionedCall2D
 Conv2D_2/StatefulPartitionedCall Conv2D_2/StatefulPartitionedCall2D
 Conv2D_3/StatefulPartitionedCall Conv2D_3/StatefulPartitionedCall2D
 Conv2D_4/StatefulPartitionedCall Conv2D_4/StatefulPartitionedCall2D
 Conv2D_5/StatefulPartitionedCall Conv2D_5/StatefulPartitionedCall2N
%Dense_Layer_6/StatefulPartitionedCall%Dense_Layer_6/StatefulPartitionedCall2N
%Dense_Log_Var/StatefulPartitionedCall%Dense_Log_Var/StatefulPartitionedCall2D
 Dense_MU/StatefulPartitionedCall Dense_MU/StatefulPartitionedCall:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs
π
ΐ
%__inference_BN_1_layer_call_fn_401680

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_1_layer_call_and_return_conditional_losses_399905
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
 
m
@__inference_Code_layer_call_and_return_conditional_losses_400492

inputs
inputs_1
identity;
ShapeShapeinputs*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*(
_output_shapes
:?????????*
dtype0
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*(
_output_shapes
:?????????}
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*(
_output_shapes
:?????????J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?W
mulMulinputs_1mul/y:output:0*
T0*(
_output_shapes
:?????????F
ExpExpmul:z:0*
T0*(
_output_shapes
:?????????[
mul_1MulExp:y:0random_normal:z:0*
T0*(
_output_shapes
:?????????R
addAddV2inputs	mul_1:z:0*
T0*(
_output_shapes
:?????????P
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Λ

@__inference_BN_4_layer_call_and_return_conditional_losses_401944

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ν
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ς
ΐ
%__inference_BN_2_layer_call_fn_401749

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_2_layer_call_and_return_conditional_losses_399938
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs

ύ
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_400190

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs
Λ

@__inference_BN_3_layer_call_and_return_conditional_losses_401862

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ν
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
²
D
(__inference_Flatten_layer_call_fn_402049

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Flatten_layer_call_and_return_conditional_losses_400315a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
ψ
D__inference_Dense_MU_layer_call_and_return_conditional_losses_402094

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ΏH

__inference__traced_save_402290
file_prefix.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop)
%savev2_bn_1_gamma_read_readvariableop(
$savev2_bn_1_beta_read_readvariableop/
+savev2_bn_1_moving_mean_read_readvariableop3
/savev2_bn_1_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop)
%savev2_bn_2_gamma_read_readvariableop(
$savev2_bn_2_beta_read_readvariableop/
+savev2_bn_2_moving_mean_read_readvariableop3
/savev2_bn_2_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop)
%savev2_bn_3_gamma_read_readvariableop(
$savev2_bn_3_beta_read_readvariableop/
+savev2_bn_3_moving_mean_read_readvariableop3
/savev2_bn_3_moving_variance_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop)
%savev2_bn_4_gamma_read_readvariableop(
$savev2_bn_4_beta_read_readvariableop/
+savev2_bn_4_moving_mean_read_readvariableop3
/savev2_bn_4_moving_variance_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop)
%savev2_bn_5_gamma_read_readvariableop(
$savev2_bn_5_beta_read_readvariableop/
+savev2_bn_5_moving_mean_read_readvariableop3
/savev2_bn_5_moving_variance_read_readvariableop3
/savev2_dense_layer_6_kernel_read_readvariableop1
-savev2_dense_layer_6_bias_read_readvariableop.
*savev2_dense_mu_kernel_read_readvariableop,
(savev2_dense_mu_bias_read_readvariableop3
/savev2_dense_log_var_kernel_read_readvariableop1
-savev2_dense_log_var_bias_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*ΐ
valueΆB³%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH·
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ο
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop%savev2_bn_1_gamma_read_readvariableop$savev2_bn_1_beta_read_readvariableop+savev2_bn_1_moving_mean_read_readvariableop/savev2_bn_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop%savev2_bn_2_gamma_read_readvariableop$savev2_bn_2_beta_read_readvariableop+savev2_bn_2_moving_mean_read_readvariableop/savev2_bn_2_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop%savev2_bn_3_gamma_read_readvariableop$savev2_bn_3_beta_read_readvariableop+savev2_bn_3_moving_mean_read_readvariableop/savev2_bn_3_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop%savev2_bn_4_gamma_read_readvariableop$savev2_bn_4_beta_read_readvariableop+savev2_bn_4_moving_mean_read_readvariableop/savev2_bn_4_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop%savev2_bn_5_gamma_read_readvariableop$savev2_bn_5_beta_read_readvariableop+savev2_bn_5_moving_mean_read_readvariableop/savev2_bn_5_moving_variance_read_readvariableop/savev2_dense_layer_6_kernel_read_readvariableop-savev2_dense_layer_6_bias_read_readvariableop*savev2_dense_mu_kernel_read_readvariableop(savev2_dense_mu_bias_read_readvariableop/savev2_dense_log_var_kernel_read_readvariableop-savev2_dense_log_var_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Φ
_input_shapesΔ
Α: : : : : : : : @:@:@:@:@:@:@::::::::::::::::::
@::
::
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
@:! 

_output_shapes	
::&!"
 
_output_shapes
:
:!"

_output_shapes	
::&#"
 
_output_shapes
:
:!$

_output_shapes	
::%

_output_shapes
: 
ϊ
Δ
%__inference_BN_5_layer_call_fn_401995

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_5_layer_call_and_return_conditional_losses_400130
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Μ

)__inference_Dense_MU_layer_call_fn_402084

inputs
unknown:

	unknown_0:	
identity’StatefulPartitionedCallέ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Dense_MU_layer_call_and_return_conditional_losses_400344p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ν

)__inference_Conv2D_2_layer_call_fn_401725

inputs!
unknown: @
	unknown_0:@
identity’StatefulPartitionedCallδ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_2_layer_call_and_return_conditional_losses_400216w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
Φ

.__inference_Dense_Layer_6_layer_call_fn_402064

inputs
unknown:
@
	unknown_0:	
identity’StatefulPartitionedCallβ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_Dense_Layer_6_layer_call_and_return_conditional_losses_400328p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????@
 
_user_specified_nameinputs
 
m
@__inference_Code_layer_call_and_return_conditional_losses_400382

inputs
inputs_1
identity;
ShapeShapeinputs*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*(
_output_shapes
:?????????*
dtype0
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*(
_output_shapes
:?????????}
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*(
_output_shapes
:?????????J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?W
mulMulinputs_1mul/y:output:0*
T0*(
_output_shapes
:?????????F
ExpExpmul:z:0*
T0*(
_output_shapes
:?????????[
mul_1MulExp:y:0random_normal:z:0*
T0*(
_output_shapes
:?????????R
addAddV2inputs	mul_1:z:0*
T0*(
_output_shapes
:?????????P
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ͺ
o
@__inference_Code_layer_call_and_return_conditional_losses_402157
inputs_0
inputs_1
identity=
ShapeShapeinputs_0*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*(
_output_shapes
:?????????*
dtype0
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*(
_output_shapes
:?????????}
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*(
_output_shapes
:?????????J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?W
mulMulinputs_1mul/y:output:0*
T0*(
_output_shapes
:?????????F
ExpExpmul:z:0*
T0*(
_output_shapes
:?????????[
mul_1MulExp:y:0random_normal:z:0*
T0*(
_output_shapes
:?????????T
addAddV2inputs_0	mul_1:z:0*
T0*(
_output_shapes
:?????????P
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????:?????????:R N
(
_output_shapes
:?????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:?????????
"
_user_specified_name
inputs/1
Λ

@__inference_BN_5_layer_call_and_return_conditional_losses_400130

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ν
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
€―
©
C__inference_encoder_layer_call_and_return_conditional_losses_401634

inputsA
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: *
bn_1_readvariableop_resource: ,
bn_1_readvariableop_1_resource: ;
-bn_1_fusedbatchnormv3_readvariableop_resource: =
/bn_1_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@*
bn_2_readvariableop_resource:@,
bn_2_readvariableop_1_resource:@;
-bn_2_fusedbatchnormv3_readvariableop_resource:@=
/bn_2_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_3_conv2d_readvariableop_resource:@7
(conv2d_3_biasadd_readvariableop_resource:	+
bn_3_readvariableop_resource:	-
bn_3_readvariableop_1_resource:	<
-bn_3_fusedbatchnormv3_readvariableop_resource:	>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_4_conv2d_readvariableop_resource:7
(conv2d_4_biasadd_readvariableop_resource:	+
bn_4_readvariableop_resource:	-
bn_4_readvariableop_1_resource:	<
-bn_4_fusedbatchnormv3_readvariableop_resource:	>
/bn_4_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_5_conv2d_readvariableop_resource:7
(conv2d_5_biasadd_readvariableop_resource:	+
bn_5_readvariableop_resource:	-
bn_5_readvariableop_1_resource:	<
-bn_5_fusedbatchnormv3_readvariableop_resource:	>
/bn_5_fusedbatchnormv3_readvariableop_1_resource:	@
,dense_layer_6_matmul_readvariableop_resource:
@<
-dense_layer_6_biasadd_readvariableop_resource:	;
'dense_mu_matmul_readvariableop_resource:
7
(dense_mu_biasadd_readvariableop_resource:	@
,dense_log_var_matmul_readvariableop_resource:
<
-dense_log_var_biasadd_readvariableop_resource:	
identity

identity_1

identity_2’BN_1/AssignNewValue’BN_1/AssignNewValue_1’$BN_1/FusedBatchNormV3/ReadVariableOp’&BN_1/FusedBatchNormV3/ReadVariableOp_1’BN_1/ReadVariableOp’BN_1/ReadVariableOp_1’BN_2/AssignNewValue’BN_2/AssignNewValue_1’$BN_2/FusedBatchNormV3/ReadVariableOp’&BN_2/FusedBatchNormV3/ReadVariableOp_1’BN_2/ReadVariableOp’BN_2/ReadVariableOp_1’BN_3/AssignNewValue’BN_3/AssignNewValue_1’$BN_3/FusedBatchNormV3/ReadVariableOp’&BN_3/FusedBatchNormV3/ReadVariableOp_1’BN_3/ReadVariableOp’BN_3/ReadVariableOp_1’BN_4/AssignNewValue’BN_4/AssignNewValue_1’$BN_4/FusedBatchNormV3/ReadVariableOp’&BN_4/FusedBatchNormV3/ReadVariableOp_1’BN_4/ReadVariableOp’BN_4/ReadVariableOp_1’BN_5/AssignNewValue’BN_5/AssignNewValue_1’$BN_5/FusedBatchNormV3/ReadVariableOp’&BN_5/FusedBatchNormV3/ReadVariableOp_1’BN_5/ReadVariableOp’BN_5/ReadVariableOp_1’Conv2D_1/BiasAdd/ReadVariableOp’Conv2D_1/Conv2D/ReadVariableOp’Conv2D_2/BiasAdd/ReadVariableOp’Conv2D_2/Conv2D/ReadVariableOp’Conv2D_3/BiasAdd/ReadVariableOp’Conv2D_3/Conv2D/ReadVariableOp’Conv2D_4/BiasAdd/ReadVariableOp’Conv2D_4/Conv2D/ReadVariableOp’Conv2D_5/BiasAdd/ReadVariableOp’Conv2D_5/Conv2D/ReadVariableOp’$Dense_Layer_6/BiasAdd/ReadVariableOp’#Dense_Layer_6/MatMul/ReadVariableOp’$Dense_Log_Var/BiasAdd/ReadVariableOp’#Dense_Log_Var/MatMul/ReadVariableOp’Dense_MU/BiasAdd/ReadVariableOp’Dense_MU/MatMul/ReadVariableOp
Conv2D_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0«
Conv2D_1/Conv2DConv2Dinputs&Conv2D_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides

Conv2D_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
Conv2D_1/BiasAddBiasAddConv2D_1/Conv2D:output:0'Conv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ j
Conv2D_1/ReluReluConv2D_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ l
BN_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes
: *
dtype0p
BN_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes
: *
dtype0
$BN_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
&BN_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ς
BN_1/FusedBatchNormV3FusedBatchNormV3Conv2D_1/Relu:activations:0BN_1/ReadVariableOp:value:0BN_1/ReadVariableOp_1:value:0,BN_1/FusedBatchNormV3/ReadVariableOp:value:0.BN_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<Ϊ
BN_1/AssignNewValueAssignVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource"BN_1/FusedBatchNormV3:batch_mean:0%^BN_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(δ
BN_1/AssignNewValue_1AssignVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource&BN_1/FusedBatchNormV3:batch_variance:0'^BN_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
Conv2D_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ύ
Conv2D_2/Conv2DConv2DBN_1/FusedBatchNormV3:y:0&Conv2D_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides

Conv2D_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
Conv2D_2/BiasAddBiasAddConv2D_2/Conv2D:output:0'Conv2D_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @j
Conv2D_2/ReluReluConv2D_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @l
BN_2/ReadVariableOpReadVariableOpbn_2_readvariableop_resource*
_output_shapes
:@*
dtype0p
BN_2/ReadVariableOp_1ReadVariableOpbn_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0
$BN_2/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
&BN_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ς
BN_2/FusedBatchNormV3FusedBatchNormV3Conv2D_2/Relu:activations:0BN_2/ReadVariableOp:value:0BN_2/ReadVariableOp_1:value:0,BN_2/FusedBatchNormV3/ReadVariableOp:value:0.BN_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<Ϊ
BN_2/AssignNewValueAssignVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource"BN_2/FusedBatchNormV3:batch_mean:0%^BN_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(δ
BN_2/AssignNewValue_1AssignVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource&BN_2/FusedBatchNormV3:batch_variance:0'^BN_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
Conv2D_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ώ
Conv2D_3/Conv2DConv2DBN_2/FusedBatchNormV3:y:0&Conv2D_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

Conv2D_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
Conv2D_3/BiasAddBiasAddConv2D_3/Conv2D:output:0'Conv2D_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????k
Conv2D_3/ReluReluConv2D_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????m
BN_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:*
dtype0q
BN_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
$BN_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
&BN_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0χ
BN_3/FusedBatchNormV3FusedBatchNormV3Conv2D_3/Relu:activations:0BN_3/ReadVariableOp:value:0BN_3/ReadVariableOp_1:value:0,BN_3/FusedBatchNormV3/ReadVariableOp:value:0.BN_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<Ϊ
BN_3/AssignNewValueAssignVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource"BN_3/FusedBatchNormV3:batch_mean:0%^BN_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(δ
BN_3/AssignNewValue_1AssignVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource&BN_3/FusedBatchNormV3:batch_variance:0'^BN_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
Conv2D_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ώ
Conv2D_4/Conv2DConv2DBN_3/FusedBatchNormV3:y:0&Conv2D_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

Conv2D_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
Conv2D_4/BiasAddBiasAddConv2D_4/Conv2D:output:0'Conv2D_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????k
Conv2D_4/ReluReluConv2D_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????m
BN_4/ReadVariableOpReadVariableOpbn_4_readvariableop_resource*
_output_shapes	
:*
dtype0q
BN_4/ReadVariableOp_1ReadVariableOpbn_4_readvariableop_1_resource*
_output_shapes	
:*
dtype0
$BN_4/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
&BN_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0χ
BN_4/FusedBatchNormV3FusedBatchNormV3Conv2D_4/Relu:activations:0BN_4/ReadVariableOp:value:0BN_4/ReadVariableOp_1:value:0,BN_4/FusedBatchNormV3/ReadVariableOp:value:0.BN_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<Ϊ
BN_4/AssignNewValueAssignVariableOp-bn_4_fusedbatchnormv3_readvariableop_resource"BN_4/FusedBatchNormV3:batch_mean:0%^BN_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(δ
BN_4/AssignNewValue_1AssignVariableOp/bn_4_fusedbatchnormv3_readvariableop_1_resource&BN_4/FusedBatchNormV3:batch_variance:0'^BN_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
Conv2D_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ώ
Conv2D_5/Conv2DConv2DBN_4/FusedBatchNormV3:y:0&Conv2D_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

Conv2D_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
Conv2D_5/BiasAddBiasAddConv2D_5/Conv2D:output:0'Conv2D_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????k
Conv2D_5/ReluReluConv2D_5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????m
BN_5/ReadVariableOpReadVariableOpbn_5_readvariableop_resource*
_output_shapes	
:*
dtype0q
BN_5/ReadVariableOp_1ReadVariableOpbn_5_readvariableop_1_resource*
_output_shapes	
:*
dtype0
$BN_5/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
&BN_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0χ
BN_5/FusedBatchNormV3FusedBatchNormV3Conv2D_5/Relu:activations:0BN_5/ReadVariableOp:value:0BN_5/ReadVariableOp_1:value:0,BN_5/FusedBatchNormV3/ReadVariableOp:value:0.BN_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<Ϊ
BN_5/AssignNewValueAssignVariableOp-bn_5_fusedbatchnormv3_readvariableop_resource"BN_5/FusedBatchNormV3:batch_mean:0%^BN_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(δ
BN_5/AssignNewValue_1AssignVariableOp/bn_5_fusedbatchnormv3_readvariableop_1_resource&BN_5/FusedBatchNormV3:batch_variance:0'^BN_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(^
Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    
Flatten/ReshapeReshapeBN_5/FusedBatchNormV3:y:0Flatten/Const:output:0*
T0*(
_output_shapes
:?????????@
#Dense_Layer_6/MatMul/ReadVariableOpReadVariableOp,dense_layer_6_matmul_readvariableop_resource* 
_output_shapes
:
@*
dtype0
Dense_Layer_6/MatMulMatMulFlatten/Reshape:output:0+Dense_Layer_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$Dense_Layer_6/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0‘
Dense_Layer_6/BiasAddBiasAddDense_Layer_6/MatMul:product:0,Dense_Layer_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????m
Dense_Layer_6/ReluReluDense_Layer_6/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
Dense_MU/MatMul/ReadVariableOpReadVariableOp'dense_mu_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
Dense_MU/MatMulMatMul Dense_Layer_6/Relu:activations:0&Dense_MU/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
Dense_MU/BiasAdd/ReadVariableOpReadVariableOp(dense_mu_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
Dense_MU/BiasAddBiasAddDense_MU/MatMul:product:0'Dense_MU/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
#Dense_Log_Var/MatMul/ReadVariableOpReadVariableOp,dense_log_var_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0 
Dense_Log_Var/MatMulMatMul Dense_Layer_6/Relu:activations:0+Dense_Log_Var/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
$Dense_Log_Var/BiasAdd/ReadVariableOpReadVariableOp-dense_log_var_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0‘
Dense_Log_Var/BiasAddBiasAddDense_Log_Var/MatMul:product:0,Dense_Log_Var/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????S

Code/ShapeShapeDense_MU/BiasAdd:output:0*
T0*
_output_shapes
:\
Code/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
Code/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
'Code/random_normal/RandomStandardNormalRandomStandardNormalCode/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0¦
Code/random_normal/mulMul0Code/random_normal/RandomStandardNormal:output:0"Code/random_normal/stddev:output:0*
T0*(
_output_shapes
:?????????
Code/random_normalAddV2Code/random_normal/mul:z:0 Code/random_normal/mean:output:0*
T0*(
_output_shapes
:?????????O

Code/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?w
Code/mulMulDense_Log_Var/BiasAdd:output:0Code/mul/y:output:0*
T0*(
_output_shapes
:?????????P
Code/ExpExpCode/mul:z:0*
T0*(
_output_shapes
:?????????j

Code/mul_1MulCode/Exp:y:0Code/random_normal:z:0*
T0*(
_output_shapes
:?????????o
Code/addAddV2Dense_MU/BiasAdd:output:0Code/mul_1:z:0*
T0*(
_output_shapes
:?????????i
IdentityIdentityDense_MU/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:?????????p

Identity_1IdentityDense_Log_Var/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:?????????^

Identity_2IdentityCode/add:z:0^NoOp*
T0*(
_output_shapes
:?????????Ξ
NoOpNoOp^BN_1/AssignNewValue^BN_1/AssignNewValue_1%^BN_1/FusedBatchNormV3/ReadVariableOp'^BN_1/FusedBatchNormV3/ReadVariableOp_1^BN_1/ReadVariableOp^BN_1/ReadVariableOp_1^BN_2/AssignNewValue^BN_2/AssignNewValue_1%^BN_2/FusedBatchNormV3/ReadVariableOp'^BN_2/FusedBatchNormV3/ReadVariableOp_1^BN_2/ReadVariableOp^BN_2/ReadVariableOp_1^BN_3/AssignNewValue^BN_3/AssignNewValue_1%^BN_3/FusedBatchNormV3/ReadVariableOp'^BN_3/FusedBatchNormV3/ReadVariableOp_1^BN_3/ReadVariableOp^BN_3/ReadVariableOp_1^BN_4/AssignNewValue^BN_4/AssignNewValue_1%^BN_4/FusedBatchNormV3/ReadVariableOp'^BN_4/FusedBatchNormV3/ReadVariableOp_1^BN_4/ReadVariableOp^BN_4/ReadVariableOp_1^BN_5/AssignNewValue^BN_5/AssignNewValue_1%^BN_5/FusedBatchNormV3/ReadVariableOp'^BN_5/FusedBatchNormV3/ReadVariableOp_1^BN_5/ReadVariableOp^BN_5/ReadVariableOp_1 ^Conv2D_1/BiasAdd/ReadVariableOp^Conv2D_1/Conv2D/ReadVariableOp ^Conv2D_2/BiasAdd/ReadVariableOp^Conv2D_2/Conv2D/ReadVariableOp ^Conv2D_3/BiasAdd/ReadVariableOp^Conv2D_3/Conv2D/ReadVariableOp ^Conv2D_4/BiasAdd/ReadVariableOp^Conv2D_4/Conv2D/ReadVariableOp ^Conv2D_5/BiasAdd/ReadVariableOp^Conv2D_5/Conv2D/ReadVariableOp%^Dense_Layer_6/BiasAdd/ReadVariableOp$^Dense_Layer_6/MatMul/ReadVariableOp%^Dense_Log_Var/BiasAdd/ReadVariableOp$^Dense_Log_Var/MatMul/ReadVariableOp ^Dense_MU/BiasAdd/ReadVariableOp^Dense_MU/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
BN_1/AssignNewValueBN_1/AssignNewValue2.
BN_1/AssignNewValue_1BN_1/AssignNewValue_12L
$BN_1/FusedBatchNormV3/ReadVariableOp$BN_1/FusedBatchNormV3/ReadVariableOp2P
&BN_1/FusedBatchNormV3/ReadVariableOp_1&BN_1/FusedBatchNormV3/ReadVariableOp_12*
BN_1/ReadVariableOpBN_1/ReadVariableOp2.
BN_1/ReadVariableOp_1BN_1/ReadVariableOp_12*
BN_2/AssignNewValueBN_2/AssignNewValue2.
BN_2/AssignNewValue_1BN_2/AssignNewValue_12L
$BN_2/FusedBatchNormV3/ReadVariableOp$BN_2/FusedBatchNormV3/ReadVariableOp2P
&BN_2/FusedBatchNormV3/ReadVariableOp_1&BN_2/FusedBatchNormV3/ReadVariableOp_12*
BN_2/ReadVariableOpBN_2/ReadVariableOp2.
BN_2/ReadVariableOp_1BN_2/ReadVariableOp_12*
BN_3/AssignNewValueBN_3/AssignNewValue2.
BN_3/AssignNewValue_1BN_3/AssignNewValue_12L
$BN_3/FusedBatchNormV3/ReadVariableOp$BN_3/FusedBatchNormV3/ReadVariableOp2P
&BN_3/FusedBatchNormV3/ReadVariableOp_1&BN_3/FusedBatchNormV3/ReadVariableOp_12*
BN_3/ReadVariableOpBN_3/ReadVariableOp2.
BN_3/ReadVariableOp_1BN_3/ReadVariableOp_12*
BN_4/AssignNewValueBN_4/AssignNewValue2.
BN_4/AssignNewValue_1BN_4/AssignNewValue_12L
$BN_4/FusedBatchNormV3/ReadVariableOp$BN_4/FusedBatchNormV3/ReadVariableOp2P
&BN_4/FusedBatchNormV3/ReadVariableOp_1&BN_4/FusedBatchNormV3/ReadVariableOp_12*
BN_4/ReadVariableOpBN_4/ReadVariableOp2.
BN_4/ReadVariableOp_1BN_4/ReadVariableOp_12*
BN_5/AssignNewValueBN_5/AssignNewValue2.
BN_5/AssignNewValue_1BN_5/AssignNewValue_12L
$BN_5/FusedBatchNormV3/ReadVariableOp$BN_5/FusedBatchNormV3/ReadVariableOp2P
&BN_5/FusedBatchNormV3/ReadVariableOp_1&BN_5/FusedBatchNormV3/ReadVariableOp_12*
BN_5/ReadVariableOpBN_5/ReadVariableOp2.
BN_5/ReadVariableOp_1BN_5/ReadVariableOp_12B
Conv2D_1/BiasAdd/ReadVariableOpConv2D_1/BiasAdd/ReadVariableOp2@
Conv2D_1/Conv2D/ReadVariableOpConv2D_1/Conv2D/ReadVariableOp2B
Conv2D_2/BiasAdd/ReadVariableOpConv2D_2/BiasAdd/ReadVariableOp2@
Conv2D_2/Conv2D/ReadVariableOpConv2D_2/Conv2D/ReadVariableOp2B
Conv2D_3/BiasAdd/ReadVariableOpConv2D_3/BiasAdd/ReadVariableOp2@
Conv2D_3/Conv2D/ReadVariableOpConv2D_3/Conv2D/ReadVariableOp2B
Conv2D_4/BiasAdd/ReadVariableOpConv2D_4/BiasAdd/ReadVariableOp2@
Conv2D_4/Conv2D/ReadVariableOpConv2D_4/Conv2D/ReadVariableOp2B
Conv2D_5/BiasAdd/ReadVariableOpConv2D_5/BiasAdd/ReadVariableOp2@
Conv2D_5/Conv2D/ReadVariableOpConv2D_5/Conv2D/ReadVariableOp2L
$Dense_Layer_6/BiasAdd/ReadVariableOp$Dense_Layer_6/BiasAdd/ReadVariableOp2J
#Dense_Layer_6/MatMul/ReadVariableOp#Dense_Layer_6/MatMul/ReadVariableOp2L
$Dense_Log_Var/BiasAdd/ReadVariableOp$Dense_Log_Var/BiasAdd/ReadVariableOp2J
#Dense_Log_Var/MatMul/ReadVariableOp#Dense_Log_Var/MatMul/ReadVariableOp2B
Dense_MU/BiasAdd/ReadVariableOpDense_MU/BiasAdd/ReadVariableOp2@
Dense_MU/MatMul/ReadVariableOpDense_MU/MatMul/ReadVariableOp:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs
ψ
Δ
%__inference_BN_4_layer_call_fn_401926

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_4_layer_call_and_return_conditional_losses_400097
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
»

@__inference_BN_2_layer_call_and_return_conditional_losses_401780

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Θ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
Η
_
C__inference_Flatten_layer_call_and_return_conditional_losses_400315

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????@Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ͺ
	
(__inference_encoder_layer_call_fn_401348

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:
@

unknown_30:	

unknown_31:


unknown_32:	

unknown_33:


unknown_34:	
identity

identity_1

identity_2’StatefulPartitionedCallΟ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:?????????:?????????:?????????*<
_read_only_resource_inputs
	
 !"#$*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_400757p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:?????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs
­
θ
!__inference__wrapped_model_399852
encoder_inputI
/encoder_conv2d_1_conv2d_readvariableop_resource: >
0encoder_conv2d_1_biasadd_readvariableop_resource: 2
$encoder_bn_1_readvariableop_resource: 4
&encoder_bn_1_readvariableop_1_resource: C
5encoder_bn_1_fusedbatchnormv3_readvariableop_resource: E
7encoder_bn_1_fusedbatchnormv3_readvariableop_1_resource: I
/encoder_conv2d_2_conv2d_readvariableop_resource: @>
0encoder_conv2d_2_biasadd_readvariableop_resource:@2
$encoder_bn_2_readvariableop_resource:@4
&encoder_bn_2_readvariableop_1_resource:@C
5encoder_bn_2_fusedbatchnormv3_readvariableop_resource:@E
7encoder_bn_2_fusedbatchnormv3_readvariableop_1_resource:@J
/encoder_conv2d_3_conv2d_readvariableop_resource:@?
0encoder_conv2d_3_biasadd_readvariableop_resource:	3
$encoder_bn_3_readvariableop_resource:	5
&encoder_bn_3_readvariableop_1_resource:	D
5encoder_bn_3_fusedbatchnormv3_readvariableop_resource:	F
7encoder_bn_3_fusedbatchnormv3_readvariableop_1_resource:	K
/encoder_conv2d_4_conv2d_readvariableop_resource:?
0encoder_conv2d_4_biasadd_readvariableop_resource:	3
$encoder_bn_4_readvariableop_resource:	5
&encoder_bn_4_readvariableop_1_resource:	D
5encoder_bn_4_fusedbatchnormv3_readvariableop_resource:	F
7encoder_bn_4_fusedbatchnormv3_readvariableop_1_resource:	K
/encoder_conv2d_5_conv2d_readvariableop_resource:?
0encoder_conv2d_5_biasadd_readvariableop_resource:	3
$encoder_bn_5_readvariableop_resource:	5
&encoder_bn_5_readvariableop_1_resource:	D
5encoder_bn_5_fusedbatchnormv3_readvariableop_resource:	F
7encoder_bn_5_fusedbatchnormv3_readvariableop_1_resource:	H
4encoder_dense_layer_6_matmul_readvariableop_resource:
@D
5encoder_dense_layer_6_biasadd_readvariableop_resource:	C
/encoder_dense_mu_matmul_readvariableop_resource:
?
0encoder_dense_mu_biasadd_readvariableop_resource:	H
4encoder_dense_log_var_matmul_readvariableop_resource:
D
5encoder_dense_log_var_biasadd_readvariableop_resource:	
identity

identity_1

identity_2’,encoder/BN_1/FusedBatchNormV3/ReadVariableOp’.encoder/BN_1/FusedBatchNormV3/ReadVariableOp_1’encoder/BN_1/ReadVariableOp’encoder/BN_1/ReadVariableOp_1’,encoder/BN_2/FusedBatchNormV3/ReadVariableOp’.encoder/BN_2/FusedBatchNormV3/ReadVariableOp_1’encoder/BN_2/ReadVariableOp’encoder/BN_2/ReadVariableOp_1’,encoder/BN_3/FusedBatchNormV3/ReadVariableOp’.encoder/BN_3/FusedBatchNormV3/ReadVariableOp_1’encoder/BN_3/ReadVariableOp’encoder/BN_3/ReadVariableOp_1’,encoder/BN_4/FusedBatchNormV3/ReadVariableOp’.encoder/BN_4/FusedBatchNormV3/ReadVariableOp_1’encoder/BN_4/ReadVariableOp’encoder/BN_4/ReadVariableOp_1’,encoder/BN_5/FusedBatchNormV3/ReadVariableOp’.encoder/BN_5/FusedBatchNormV3/ReadVariableOp_1’encoder/BN_5/ReadVariableOp’encoder/BN_5/ReadVariableOp_1’'encoder/Conv2D_1/BiasAdd/ReadVariableOp’&encoder/Conv2D_1/Conv2D/ReadVariableOp’'encoder/Conv2D_2/BiasAdd/ReadVariableOp’&encoder/Conv2D_2/Conv2D/ReadVariableOp’'encoder/Conv2D_3/BiasAdd/ReadVariableOp’&encoder/Conv2D_3/Conv2D/ReadVariableOp’'encoder/Conv2D_4/BiasAdd/ReadVariableOp’&encoder/Conv2D_4/Conv2D/ReadVariableOp’'encoder/Conv2D_5/BiasAdd/ReadVariableOp’&encoder/Conv2D_5/Conv2D/ReadVariableOp’,encoder/Dense_Layer_6/BiasAdd/ReadVariableOp’+encoder/Dense_Layer_6/MatMul/ReadVariableOp’,encoder/Dense_Log_Var/BiasAdd/ReadVariableOp’+encoder/Dense_Log_Var/MatMul/ReadVariableOp’'encoder/Dense_MU/BiasAdd/ReadVariableOp’&encoder/Dense_MU/MatMul/ReadVariableOp
&encoder/Conv2D_1/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Β
encoder/Conv2D_1/Conv2DConv2Dencoder_input.encoder/Conv2D_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides

'encoder/Conv2D_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0°
encoder/Conv2D_1/BiasAddBiasAdd encoder/Conv2D_1/Conv2D:output:0/encoder/Conv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ z
encoder/Conv2D_1/ReluRelu!encoder/Conv2D_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ |
encoder/BN_1/ReadVariableOpReadVariableOp$encoder_bn_1_readvariableop_resource*
_output_shapes
: *
dtype0
encoder/BN_1/ReadVariableOp_1ReadVariableOp&encoder_bn_1_readvariableop_1_resource*
_output_shapes
: *
dtype0
,encoder/BN_1/FusedBatchNormV3/ReadVariableOpReadVariableOp5encoder_bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0’
.encoder/BN_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7encoder_bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0
encoder/BN_1/FusedBatchNormV3FusedBatchNormV3#encoder/Conv2D_1/Relu:activations:0#encoder/BN_1/ReadVariableOp:value:0%encoder/BN_1/ReadVariableOp_1:value:04encoder/BN_1/FusedBatchNormV3/ReadVariableOp:value:06encoder/BN_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o:*
is_training( 
&encoder/Conv2D_2/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Φ
encoder/Conv2D_2/Conv2DConv2D!encoder/BN_1/FusedBatchNormV3:y:0.encoder/Conv2D_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides

'encoder/Conv2D_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0°
encoder/Conv2D_2/BiasAddBiasAdd encoder/Conv2D_2/Conv2D:output:0/encoder/Conv2D_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @z
encoder/Conv2D_2/ReluRelu!encoder/Conv2D_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @|
encoder/BN_2/ReadVariableOpReadVariableOp$encoder_bn_2_readvariableop_resource*
_output_shapes
:@*
dtype0
encoder/BN_2/ReadVariableOp_1ReadVariableOp&encoder_bn_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0
,encoder/BN_2/FusedBatchNormV3/ReadVariableOpReadVariableOp5encoder_bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0’
.encoder/BN_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7encoder_bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
encoder/BN_2/FusedBatchNormV3FusedBatchNormV3#encoder/Conv2D_2/Relu:activations:0#encoder/BN_2/ReadVariableOp:value:0%encoder/BN_2/ReadVariableOp_1:value:04encoder/BN_2/FusedBatchNormV3/ReadVariableOp:value:06encoder/BN_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o:*
is_training( 
&encoder/Conv2D_3/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Χ
encoder/Conv2D_3/Conv2DConv2D!encoder/BN_2/FusedBatchNormV3:y:0.encoder/Conv2D_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

'encoder/Conv2D_3/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0±
encoder/Conv2D_3/BiasAddBiasAdd encoder/Conv2D_3/Conv2D:output:0/encoder/Conv2D_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????{
encoder/Conv2D_3/ReluRelu!encoder/Conv2D_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????}
encoder/BN_3/ReadVariableOpReadVariableOp$encoder_bn_3_readvariableop_resource*
_output_shapes	
:*
dtype0
encoder/BN_3/ReadVariableOp_1ReadVariableOp&encoder_bn_3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
,encoder/BN_3/FusedBatchNormV3/ReadVariableOpReadVariableOp5encoder_bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0£
.encoder/BN_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7encoder_bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
encoder/BN_3/FusedBatchNormV3FusedBatchNormV3#encoder/Conv2D_3/Relu:activations:0#encoder/BN_3/ReadVariableOp:value:0%encoder/BN_3/ReadVariableOp_1:value:04encoder/BN_3/FusedBatchNormV3/ReadVariableOp:value:06encoder/BN_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training(  
&encoder/Conv2D_4/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Χ
encoder/Conv2D_4/Conv2DConv2D!encoder/BN_3/FusedBatchNormV3:y:0.encoder/Conv2D_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

'encoder/Conv2D_4/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0±
encoder/Conv2D_4/BiasAddBiasAdd encoder/Conv2D_4/Conv2D:output:0/encoder/Conv2D_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????{
encoder/Conv2D_4/ReluRelu!encoder/Conv2D_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????}
encoder/BN_4/ReadVariableOpReadVariableOp$encoder_bn_4_readvariableop_resource*
_output_shapes	
:*
dtype0
encoder/BN_4/ReadVariableOp_1ReadVariableOp&encoder_bn_4_readvariableop_1_resource*
_output_shapes	
:*
dtype0
,encoder/BN_4/FusedBatchNormV3/ReadVariableOpReadVariableOp5encoder_bn_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0£
.encoder/BN_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7encoder_bn_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
encoder/BN_4/FusedBatchNormV3FusedBatchNormV3#encoder/Conv2D_4/Relu:activations:0#encoder/BN_4/ReadVariableOp:value:0%encoder/BN_4/ReadVariableOp_1:value:04encoder/BN_4/FusedBatchNormV3/ReadVariableOp:value:06encoder/BN_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training(  
&encoder/Conv2D_5/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Χ
encoder/Conv2D_5/Conv2DConv2D!encoder/BN_4/FusedBatchNormV3:y:0.encoder/Conv2D_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

'encoder/Conv2D_5/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0±
encoder/Conv2D_5/BiasAddBiasAdd encoder/Conv2D_5/Conv2D:output:0/encoder/Conv2D_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????{
encoder/Conv2D_5/ReluRelu!encoder/Conv2D_5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????}
encoder/BN_5/ReadVariableOpReadVariableOp$encoder_bn_5_readvariableop_resource*
_output_shapes	
:*
dtype0
encoder/BN_5/ReadVariableOp_1ReadVariableOp&encoder_bn_5_readvariableop_1_resource*
_output_shapes	
:*
dtype0
,encoder/BN_5/FusedBatchNormV3/ReadVariableOpReadVariableOp5encoder_bn_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0£
.encoder/BN_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7encoder_bn_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
encoder/BN_5/FusedBatchNormV3FusedBatchNormV3#encoder/Conv2D_5/Relu:activations:0#encoder/BN_5/ReadVariableOp:value:0%encoder/BN_5/ReadVariableOp_1:value:04encoder/BN_5/FusedBatchNormV3/ReadVariableOp:value:06encoder/BN_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( f
encoder/Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    
encoder/Flatten/ReshapeReshape!encoder/BN_5/FusedBatchNormV3:y:0encoder/Flatten/Const:output:0*
T0*(
_output_shapes
:?????????@’
+encoder/Dense_Layer_6/MatMul/ReadVariableOpReadVariableOp4encoder_dense_layer_6_matmul_readvariableop_resource* 
_output_shapes
:
@*
dtype0°
encoder/Dense_Layer_6/MatMulMatMul encoder/Flatten/Reshape:output:03encoder/Dense_Layer_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
,encoder/Dense_Layer_6/BiasAdd/ReadVariableOpReadVariableOp5encoder_dense_layer_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ή
encoder/Dense_Layer_6/BiasAddBiasAdd&encoder/Dense_Layer_6/MatMul:product:04encoder/Dense_Layer_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????}
encoder/Dense_Layer_6/ReluRelu&encoder/Dense_Layer_6/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
&encoder/Dense_MU/MatMul/ReadVariableOpReadVariableOp/encoder_dense_mu_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0?
encoder/Dense_MU/MatMulMatMul(encoder/Dense_Layer_6/Relu:activations:0.encoder/Dense_MU/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
'encoder/Dense_MU/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_mu_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ͺ
encoder/Dense_MU/BiasAddBiasAdd!encoder/Dense_MU/MatMul:product:0/encoder/Dense_MU/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????’
+encoder/Dense_Log_Var/MatMul/ReadVariableOpReadVariableOp4encoder_dense_log_var_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Έ
encoder/Dense_Log_Var/MatMulMatMul(encoder/Dense_Layer_6/Relu:activations:03encoder/Dense_Log_Var/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
,encoder/Dense_Log_Var/BiasAdd/ReadVariableOpReadVariableOp5encoder_dense_log_var_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ή
encoder/Dense_Log_Var/BiasAddBiasAdd&encoder/Dense_Log_Var/MatMul:product:04encoder/Dense_Log_Var/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????c
encoder/Code/ShapeShape!encoder/Dense_MU/BiasAdd:output:0*
T0*
_output_shapes
:d
encoder/Code/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    f
!encoder/Code/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?€
/encoder/Code/random_normal/RandomStandardNormalRandomStandardNormalencoder/Code/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0Ύ
encoder/Code/random_normal/mulMul8encoder/Code/random_normal/RandomStandardNormal:output:0*encoder/Code/random_normal/stddev:output:0*
T0*(
_output_shapes
:?????????€
encoder/Code/random_normalAddV2"encoder/Code/random_normal/mul:z:0(encoder/Code/random_normal/mean:output:0*
T0*(
_output_shapes
:?????????W
encoder/Code/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
encoder/Code/mulMul&encoder/Dense_Log_Var/BiasAdd:output:0encoder/Code/mul/y:output:0*
T0*(
_output_shapes
:?????????`
encoder/Code/ExpExpencoder/Code/mul:z:0*
T0*(
_output_shapes
:?????????
encoder/Code/mul_1Mulencoder/Code/Exp:y:0encoder/Code/random_normal:z:0*
T0*(
_output_shapes
:?????????
encoder/Code/addAddV2!encoder/Dense_MU/BiasAdd:output:0encoder/Code/mul_1:z:0*
T0*(
_output_shapes
:?????????d
IdentityIdentityencoder/Code/add:z:0^NoOp*
T0*(
_output_shapes
:?????????x

Identity_1Identity&encoder/Dense_Log_Var/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:?????????s

Identity_2Identity!encoder/Dense_MU/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:?????????
NoOpNoOp-^encoder/BN_1/FusedBatchNormV3/ReadVariableOp/^encoder/BN_1/FusedBatchNormV3/ReadVariableOp_1^encoder/BN_1/ReadVariableOp^encoder/BN_1/ReadVariableOp_1-^encoder/BN_2/FusedBatchNormV3/ReadVariableOp/^encoder/BN_2/FusedBatchNormV3/ReadVariableOp_1^encoder/BN_2/ReadVariableOp^encoder/BN_2/ReadVariableOp_1-^encoder/BN_3/FusedBatchNormV3/ReadVariableOp/^encoder/BN_3/FusedBatchNormV3/ReadVariableOp_1^encoder/BN_3/ReadVariableOp^encoder/BN_3/ReadVariableOp_1-^encoder/BN_4/FusedBatchNormV3/ReadVariableOp/^encoder/BN_4/FusedBatchNormV3/ReadVariableOp_1^encoder/BN_4/ReadVariableOp^encoder/BN_4/ReadVariableOp_1-^encoder/BN_5/FusedBatchNormV3/ReadVariableOp/^encoder/BN_5/FusedBatchNormV3/ReadVariableOp_1^encoder/BN_5/ReadVariableOp^encoder/BN_5/ReadVariableOp_1(^encoder/Conv2D_1/BiasAdd/ReadVariableOp'^encoder/Conv2D_1/Conv2D/ReadVariableOp(^encoder/Conv2D_2/BiasAdd/ReadVariableOp'^encoder/Conv2D_2/Conv2D/ReadVariableOp(^encoder/Conv2D_3/BiasAdd/ReadVariableOp'^encoder/Conv2D_3/Conv2D/ReadVariableOp(^encoder/Conv2D_4/BiasAdd/ReadVariableOp'^encoder/Conv2D_4/Conv2D/ReadVariableOp(^encoder/Conv2D_5/BiasAdd/ReadVariableOp'^encoder/Conv2D_5/Conv2D/ReadVariableOp-^encoder/Dense_Layer_6/BiasAdd/ReadVariableOp,^encoder/Dense_Layer_6/MatMul/ReadVariableOp-^encoder/Dense_Log_Var/BiasAdd/ReadVariableOp,^encoder/Dense_Log_Var/MatMul/ReadVariableOp(^encoder/Dense_MU/BiasAdd/ReadVariableOp'^encoder/Dense_MU/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,encoder/BN_1/FusedBatchNormV3/ReadVariableOp,encoder/BN_1/FusedBatchNormV3/ReadVariableOp2`
.encoder/BN_1/FusedBatchNormV3/ReadVariableOp_1.encoder/BN_1/FusedBatchNormV3/ReadVariableOp_12:
encoder/BN_1/ReadVariableOpencoder/BN_1/ReadVariableOp2>
encoder/BN_1/ReadVariableOp_1encoder/BN_1/ReadVariableOp_12\
,encoder/BN_2/FusedBatchNormV3/ReadVariableOp,encoder/BN_2/FusedBatchNormV3/ReadVariableOp2`
.encoder/BN_2/FusedBatchNormV3/ReadVariableOp_1.encoder/BN_2/FusedBatchNormV3/ReadVariableOp_12:
encoder/BN_2/ReadVariableOpencoder/BN_2/ReadVariableOp2>
encoder/BN_2/ReadVariableOp_1encoder/BN_2/ReadVariableOp_12\
,encoder/BN_3/FusedBatchNormV3/ReadVariableOp,encoder/BN_3/FusedBatchNormV3/ReadVariableOp2`
.encoder/BN_3/FusedBatchNormV3/ReadVariableOp_1.encoder/BN_3/FusedBatchNormV3/ReadVariableOp_12:
encoder/BN_3/ReadVariableOpencoder/BN_3/ReadVariableOp2>
encoder/BN_3/ReadVariableOp_1encoder/BN_3/ReadVariableOp_12\
,encoder/BN_4/FusedBatchNormV3/ReadVariableOp,encoder/BN_4/FusedBatchNormV3/ReadVariableOp2`
.encoder/BN_4/FusedBatchNormV3/ReadVariableOp_1.encoder/BN_4/FusedBatchNormV3/ReadVariableOp_12:
encoder/BN_4/ReadVariableOpencoder/BN_4/ReadVariableOp2>
encoder/BN_4/ReadVariableOp_1encoder/BN_4/ReadVariableOp_12\
,encoder/BN_5/FusedBatchNormV3/ReadVariableOp,encoder/BN_5/FusedBatchNormV3/ReadVariableOp2`
.encoder/BN_5/FusedBatchNormV3/ReadVariableOp_1.encoder/BN_5/FusedBatchNormV3/ReadVariableOp_12:
encoder/BN_5/ReadVariableOpencoder/BN_5/ReadVariableOp2>
encoder/BN_5/ReadVariableOp_1encoder/BN_5/ReadVariableOp_12R
'encoder/Conv2D_1/BiasAdd/ReadVariableOp'encoder/Conv2D_1/BiasAdd/ReadVariableOp2P
&encoder/Conv2D_1/Conv2D/ReadVariableOp&encoder/Conv2D_1/Conv2D/ReadVariableOp2R
'encoder/Conv2D_2/BiasAdd/ReadVariableOp'encoder/Conv2D_2/BiasAdd/ReadVariableOp2P
&encoder/Conv2D_2/Conv2D/ReadVariableOp&encoder/Conv2D_2/Conv2D/ReadVariableOp2R
'encoder/Conv2D_3/BiasAdd/ReadVariableOp'encoder/Conv2D_3/BiasAdd/ReadVariableOp2P
&encoder/Conv2D_3/Conv2D/ReadVariableOp&encoder/Conv2D_3/Conv2D/ReadVariableOp2R
'encoder/Conv2D_4/BiasAdd/ReadVariableOp'encoder/Conv2D_4/BiasAdd/ReadVariableOp2P
&encoder/Conv2D_4/Conv2D/ReadVariableOp&encoder/Conv2D_4/Conv2D/ReadVariableOp2R
'encoder/Conv2D_5/BiasAdd/ReadVariableOp'encoder/Conv2D_5/BiasAdd/ReadVariableOp2P
&encoder/Conv2D_5/Conv2D/ReadVariableOp&encoder/Conv2D_5/Conv2D/ReadVariableOp2\
,encoder/Dense_Layer_6/BiasAdd/ReadVariableOp,encoder/Dense_Layer_6/BiasAdd/ReadVariableOp2Z
+encoder/Dense_Layer_6/MatMul/ReadVariableOp+encoder/Dense_Layer_6/MatMul/ReadVariableOp2\
,encoder/Dense_Log_Var/BiasAdd/ReadVariableOp,encoder/Dense_Log_Var/BiasAdd/ReadVariableOp2Z
+encoder/Dense_Log_Var/MatMul/ReadVariableOp+encoder/Dense_Log_Var/MatMul/ReadVariableOp2R
'encoder/Dense_MU/BiasAdd/ReadVariableOp'encoder/Dense_MU/BiasAdd/ReadVariableOp2P
&encoder/Dense_MU/MatMul/ReadVariableOp&encoder/Dense_MU/MatMul/ReadVariableOp:` \
1
_output_shapes
:?????????
'
_user_specified_nameEncoder_Input
Χ	
ύ
I__inference_Dense_Log_Var_layer_call_and_return_conditional_losses_402113

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
υ
―
@__inference_BN_1_layer_call_and_return_conditional_losses_401716

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity’AssignNewValue’AssignNewValue_1’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Φ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<Ζ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Π
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? Τ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
υ
―
@__inference_BN_2_layer_call_and_return_conditional_losses_401798

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity’AssignNewValue’AssignNewValue_1’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Φ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<Ζ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Π
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@Τ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
£
	
$__inference_signature_wrapper_401186
encoder_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:
@

unknown_30:	

unknown_31:


unknown_32:	

unknown_33:


unknown_34:	
identity

identity_1

identity_2’StatefulPartitionedCallΎ
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:?????????:?????????:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_399852p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:?????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
1
_output_shapes
:?????????
'
_user_specified_nameEncoder_Input
»

@__inference_BN_1_layer_call_and_return_conditional_losses_399874

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Θ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
ψ
D__inference_Dense_MU_layer_call_and_return_conditional_losses_400344

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Χ	
ύ
I__inference_Dense_Log_Var_layer_call_and_return_conditional_losses_400360

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

?
D__inference_Conv2D_3_layer_call_and_return_conditional_losses_400242

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
ϊO
κ
C__inference_encoder_layer_call_and_return_conditional_losses_401103
encoder_input)
conv2d_1_401013: 
conv2d_1_401015: 
bn_1_401018: 
bn_1_401020: 
bn_1_401022: 
bn_1_401024: )
conv2d_2_401027: @
conv2d_2_401029:@
bn_2_401032:@
bn_2_401034:@
bn_2_401036:@
bn_2_401038:@*
conv2d_3_401041:@
conv2d_3_401043:	
bn_3_401046:	
bn_3_401048:	
bn_3_401050:	
bn_3_401052:	+
conv2d_4_401055:
conv2d_4_401057:	
bn_4_401060:	
bn_4_401062:	
bn_4_401064:	
bn_4_401066:	+
conv2d_5_401069:
conv2d_5_401071:	
bn_5_401074:	
bn_5_401076:	
bn_5_401078:	
bn_5_401080:	(
dense_layer_6_401084:
@#
dense_layer_6_401086:	#
dense_mu_401089:

dense_mu_401091:	(
dense_log_var_401094:
#
dense_log_var_401096:	
identity

identity_1

identity_2’BN_1/StatefulPartitionedCall’BN_2/StatefulPartitionedCall’BN_3/StatefulPartitionedCall’BN_4/StatefulPartitionedCall’BN_5/StatefulPartitionedCall’Code/StatefulPartitionedCall’ Conv2D_1/StatefulPartitionedCall’ Conv2D_2/StatefulPartitionedCall’ Conv2D_3/StatefulPartitionedCall’ Conv2D_4/StatefulPartitionedCall’ Conv2D_5/StatefulPartitionedCall’%Dense_Layer_6/StatefulPartitionedCall’%Dense_Log_Var/StatefulPartitionedCall’ Dense_MU/StatefulPartitionedCall
 Conv2D_1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputconv2d_1_401013conv2d_1_401015*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_400190ͺ
BN_1/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_1/StatefulPartitionedCall:output:0bn_1_401018bn_1_401020bn_1_401022bn_1_401024*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_1_layer_call_and_return_conditional_losses_399905
 Conv2D_2/StatefulPartitionedCallStatefulPartitionedCall%BN_1/StatefulPartitionedCall:output:0conv2d_2_401027conv2d_2_401029*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_2_layer_call_and_return_conditional_losses_400216ͺ
BN_2/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_2/StatefulPartitionedCall:output:0bn_2_401032bn_2_401034bn_2_401036bn_2_401038*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_2_layer_call_and_return_conditional_losses_399969
 Conv2D_3/StatefulPartitionedCallStatefulPartitionedCall%BN_2/StatefulPartitionedCall:output:0conv2d_3_401041conv2d_3_401043*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_3_layer_call_and_return_conditional_losses_400242«
BN_3/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_3/StatefulPartitionedCall:output:0bn_3_401046bn_3_401048bn_3_401050bn_3_401052*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_3_layer_call_and_return_conditional_losses_400033
 Conv2D_4/StatefulPartitionedCallStatefulPartitionedCall%BN_3/StatefulPartitionedCall:output:0conv2d_4_401055conv2d_4_401057*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_4_layer_call_and_return_conditional_losses_400268«
BN_4/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_4/StatefulPartitionedCall:output:0bn_4_401060bn_4_401062bn_4_401064bn_4_401066*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_4_layer_call_and_return_conditional_losses_400097
 Conv2D_5/StatefulPartitionedCallStatefulPartitionedCall%BN_4/StatefulPartitionedCall:output:0conv2d_5_401069conv2d_5_401071*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_5_layer_call_and_return_conditional_losses_400294«
BN_5/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_5/StatefulPartitionedCall:output:0bn_5_401074bn_5_401076bn_5_401078bn_5_401080*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_5_layer_call_and_return_conditional_losses_400161Ω
Flatten/PartitionedCallPartitionedCall%BN_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Flatten_layer_call_and_return_conditional_losses_400315’
%Dense_Layer_6/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0dense_layer_6_401084dense_layer_6_401086*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_Dense_Layer_6_layer_call_and_return_conditional_losses_400328
 Dense_MU/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_6/StatefulPartitionedCall:output:0dense_mu_401089dense_mu_401091*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Dense_MU_layer_call_and_return_conditional_losses_400344°
%Dense_Log_Var/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_6/StatefulPartitionedCall:output:0dense_log_var_401094dense_log_var_401096*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_Dense_Log_Var_layer_call_and_return_conditional_losses_400360
Code/StatefulPartitionedCallStatefulPartitionedCall)Dense_MU/StatefulPartitionedCall:output:0.Dense_Log_Var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Code_layer_call_and_return_conditional_losses_400492y
IdentityIdentity)Dense_MU/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????

Identity_1Identity.Dense_Log_Var/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????w

Identity_2Identity%Code/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????’
NoOpNoOp^BN_1/StatefulPartitionedCall^BN_2/StatefulPartitionedCall^BN_3/StatefulPartitionedCall^BN_4/StatefulPartitionedCall^BN_5/StatefulPartitionedCall^Code/StatefulPartitionedCall!^Conv2D_1/StatefulPartitionedCall!^Conv2D_2/StatefulPartitionedCall!^Conv2D_3/StatefulPartitionedCall!^Conv2D_4/StatefulPartitionedCall!^Conv2D_5/StatefulPartitionedCall&^Dense_Layer_6/StatefulPartitionedCall&^Dense_Log_Var/StatefulPartitionedCall!^Dense_MU/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
BN_1/StatefulPartitionedCallBN_1/StatefulPartitionedCall2<
BN_2/StatefulPartitionedCallBN_2/StatefulPartitionedCall2<
BN_3/StatefulPartitionedCallBN_3/StatefulPartitionedCall2<
BN_4/StatefulPartitionedCallBN_4/StatefulPartitionedCall2<
BN_5/StatefulPartitionedCallBN_5/StatefulPartitionedCall2<
Code/StatefulPartitionedCallCode/StatefulPartitionedCall2D
 Conv2D_1/StatefulPartitionedCall Conv2D_1/StatefulPartitionedCall2D
 Conv2D_2/StatefulPartitionedCall Conv2D_2/StatefulPartitionedCall2D
 Conv2D_3/StatefulPartitionedCall Conv2D_3/StatefulPartitionedCall2D
 Conv2D_4/StatefulPartitionedCall Conv2D_4/StatefulPartitionedCall2D
 Conv2D_5/StatefulPartitionedCall Conv2D_5/StatefulPartitionedCall2N
%Dense_Layer_6/StatefulPartitionedCall%Dense_Layer_6/StatefulPartitionedCall2N
%Dense_Log_Var/StatefulPartitionedCall%Dense_Log_Var/StatefulPartitionedCall2D
 Dense_MU/StatefulPartitionedCall Dense_MU/StatefulPartitionedCall:` \
1
_output_shapes
:?????????
'
_user_specified_nameEncoder_Input
ς
ΐ
%__inference_BN_1_layer_call_fn_401667

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_1_layer_call_and_return_conditional_losses_399874
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
¬

ύ
I__inference_Dense_Layer_6_layer_call_and_return_conditional_losses_402075

inputs2
matmul_readvariableop_resource:
@.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
@*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????@
 
_user_specified_nameinputs

³
@__inference_BN_5_layer_call_and_return_conditional_losses_402044

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’AssignNewValue’AssignNewValue_1’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ϋ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<Ζ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Π
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????Τ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs

ύ
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_401654

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs
Ι
	
(__inference_encoder_layer_call_fn_400466
encoder_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:
@

unknown_30:	

unknown_31:


unknown_32:	

unknown_33:


unknown_34:	
identity

identity_1

identity_2’StatefulPartitionedCallΰ
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:?????????:?????????:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_400387p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:?????????r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
1
_output_shapes
:?????????
'
_user_specified_nameEncoder_Input
ρ

)__inference_Conv2D_1_layer_call_fn_401643

inputs!
unknown: 
	unknown_0: 
identity’StatefulPartitionedCallδ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_400190w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs


D__inference_Conv2D_5_layer_call_and_return_conditional_losses_400294

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
»

@__inference_BN_2_layer_call_and_return_conditional_losses_399938

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Θ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs

ύ
D__inference_Conv2D_2_layer_call_and_return_conditional_losses_400216

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????  @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
υ
―
@__inference_BN_2_layer_call_and_return_conditional_losses_399969

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity’AssignNewValue’AssignNewValue_1’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Φ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<Ζ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Π
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@Τ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
ρ
 
)__inference_Conv2D_3_layer_call_fn_401807

inputs"
unknown:@
	unknown_0:	
identity’StatefulPartitionedCallε
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_3_layer_call_and_return_conditional_losses_400242x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
π
ΐ
%__inference_BN_2_layer_call_fn_401762

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_2_layer_call_and_return_conditional_losses_399969
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
οO
γ
C__inference_encoder_layer_call_and_return_conditional_losses_400387

inputs)
conv2d_1_400191: 
conv2d_1_400193: 
bn_1_400196: 
bn_1_400198: 
bn_1_400200: 
bn_1_400202: )
conv2d_2_400217: @
conv2d_2_400219:@
bn_2_400222:@
bn_2_400224:@
bn_2_400226:@
bn_2_400228:@*
conv2d_3_400243:@
conv2d_3_400245:	
bn_3_400248:	
bn_3_400250:	
bn_3_400252:	
bn_3_400254:	+
conv2d_4_400269:
conv2d_4_400271:	
bn_4_400274:	
bn_4_400276:	
bn_4_400278:	
bn_4_400280:	+
conv2d_5_400295:
conv2d_5_400297:	
bn_5_400300:	
bn_5_400302:	
bn_5_400304:	
bn_5_400306:	(
dense_layer_6_400329:
@#
dense_layer_6_400331:	#
dense_mu_400345:

dense_mu_400347:	(
dense_log_var_400361:
#
dense_log_var_400363:	
identity

identity_1

identity_2’BN_1/StatefulPartitionedCall’BN_2/StatefulPartitionedCall’BN_3/StatefulPartitionedCall’BN_4/StatefulPartitionedCall’BN_5/StatefulPartitionedCall’Code/StatefulPartitionedCall’ Conv2D_1/StatefulPartitionedCall’ Conv2D_2/StatefulPartitionedCall’ Conv2D_3/StatefulPartitionedCall’ Conv2D_4/StatefulPartitionedCall’ Conv2D_5/StatefulPartitionedCall’%Dense_Layer_6/StatefulPartitionedCall’%Dense_Log_Var/StatefulPartitionedCall’ Dense_MU/StatefulPartitionedCallϋ
 Conv2D_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_400191conv2d_1_400193*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_400190¬
BN_1/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_1/StatefulPartitionedCall:output:0bn_1_400196bn_1_400198bn_1_400200bn_1_400202*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_1_layer_call_and_return_conditional_losses_399874
 Conv2D_2/StatefulPartitionedCallStatefulPartitionedCall%BN_1/StatefulPartitionedCall:output:0conv2d_2_400217conv2d_2_400219*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_2_layer_call_and_return_conditional_losses_400216¬
BN_2/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_2/StatefulPartitionedCall:output:0bn_2_400222bn_2_400224bn_2_400226bn_2_400228*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_2_layer_call_and_return_conditional_losses_399938
 Conv2D_3/StatefulPartitionedCallStatefulPartitionedCall%BN_2/StatefulPartitionedCall:output:0conv2d_3_400243conv2d_3_400245*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_3_layer_call_and_return_conditional_losses_400242­
BN_3/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_3/StatefulPartitionedCall:output:0bn_3_400248bn_3_400250bn_3_400252bn_3_400254*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_3_layer_call_and_return_conditional_losses_400002
 Conv2D_4/StatefulPartitionedCallStatefulPartitionedCall%BN_3/StatefulPartitionedCall:output:0conv2d_4_400269conv2d_4_400271*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_4_layer_call_and_return_conditional_losses_400268­
BN_4/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_4/StatefulPartitionedCall:output:0bn_4_400274bn_4_400276bn_4_400278bn_4_400280*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_4_layer_call_and_return_conditional_losses_400066
 Conv2D_5/StatefulPartitionedCallStatefulPartitionedCall%BN_4/StatefulPartitionedCall:output:0conv2d_5_400295conv2d_5_400297*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Conv2D_5_layer_call_and_return_conditional_losses_400294­
BN_5/StatefulPartitionedCallStatefulPartitionedCall)Conv2D_5/StatefulPartitionedCall:output:0bn_5_400300bn_5_400302bn_5_400304bn_5_400306*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_BN_5_layer_call_and_return_conditional_losses_400130Ω
Flatten/PartitionedCallPartitionedCall%BN_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Flatten_layer_call_and_return_conditional_losses_400315’
%Dense_Layer_6/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0dense_layer_6_400329dense_layer_6_400331*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_Dense_Layer_6_layer_call_and_return_conditional_losses_400328
 Dense_MU/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_6/StatefulPartitionedCall:output:0dense_mu_400345dense_mu_400347*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Dense_MU_layer_call_and_return_conditional_losses_400344°
%Dense_Log_Var/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_6/StatefulPartitionedCall:output:0dense_log_var_400361dense_log_var_400363*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_Dense_Log_Var_layer_call_and_return_conditional_losses_400360
Code/StatefulPartitionedCallStatefulPartitionedCall)Dense_MU/StatefulPartitionedCall:output:0.Dense_Log_Var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_Code_layer_call_and_return_conditional_losses_400382y
IdentityIdentity)Dense_MU/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????

Identity_1Identity.Dense_Log_Var/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????w

Identity_2Identity%Code/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????’
NoOpNoOp^BN_1/StatefulPartitionedCall^BN_2/StatefulPartitionedCall^BN_3/StatefulPartitionedCall^BN_4/StatefulPartitionedCall^BN_5/StatefulPartitionedCall^Code/StatefulPartitionedCall!^Conv2D_1/StatefulPartitionedCall!^Conv2D_2/StatefulPartitionedCall!^Conv2D_3/StatefulPartitionedCall!^Conv2D_4/StatefulPartitionedCall!^Conv2D_5/StatefulPartitionedCall&^Dense_Layer_6/StatefulPartitionedCall&^Dense_Log_Var/StatefulPartitionedCall!^Dense_MU/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
BN_1/StatefulPartitionedCallBN_1/StatefulPartitionedCall2<
BN_2/StatefulPartitionedCallBN_2/StatefulPartitionedCall2<
BN_3/StatefulPartitionedCallBN_3/StatefulPartitionedCall2<
BN_4/StatefulPartitionedCallBN_4/StatefulPartitionedCall2<
BN_5/StatefulPartitionedCallBN_5/StatefulPartitionedCall2<
Code/StatefulPartitionedCallCode/StatefulPartitionedCall2D
 Conv2D_1/StatefulPartitionedCall Conv2D_1/StatefulPartitionedCall2D
 Conv2D_2/StatefulPartitionedCall Conv2D_2/StatefulPartitionedCall2D
 Conv2D_3/StatefulPartitionedCall Conv2D_3/StatefulPartitionedCall2D
 Conv2D_4/StatefulPartitionedCall Conv2D_4/StatefulPartitionedCall2D
 Conv2D_5/StatefulPartitionedCall Conv2D_5/StatefulPartitionedCall2N
%Dense_Layer_6/StatefulPartitionedCall%Dense_Layer_6/StatefulPartitionedCall2N
%Dense_Log_Var/StatefulPartitionedCall%Dense_Log_Var/StatefulPartitionedCall2D
 Dense_MU/StatefulPartitionedCall Dense_MU/StatefulPartitionedCall:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs
Η
_
C__inference_Flatten_layer_call_and_return_conditional_losses_402055

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????@Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
¬

ύ
I__inference_Dense_Layer_6_layer_call_and_return_conditional_losses_400328

inputs2
matmul_readvariableop_resource:
@.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
@*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????@
 
_user_specified_nameinputs

³
@__inference_BN_5_layer_call_and_return_conditional_losses_400161

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’AssignNewValue’AssignNewValue_1’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ϋ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<Ζ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Π
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????Τ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
υ
―
@__inference_BN_1_layer_call_and_return_conditional_losses_399905

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity’AssignNewValue’AssignNewValue_1’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Φ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<Ζ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Π
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? Τ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs

³
@__inference_BN_4_layer_call_and_return_conditional_losses_401962

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity’AssignNewValue’AssignNewValue_1’FusedBatchNormV3/ReadVariableOp’!FusedBatchNormV3/ReadVariableOp_1’ReadVariableOp’ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ϋ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<Ζ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Π
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????Τ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs"ΏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Α
serving_default­
Q
Encoder_Input@
serving_default_Encoder_Input:0?????????9
Code1
StatefulPartitionedCall:0?????????B
Dense_Log_Var1
StatefulPartitionedCall:1?????????=
Dense_MU1
StatefulPartitionedCall:2?????????tensorflow/serving/predict:
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
layer_with_weights-12
layer-14
layer-15
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
έ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias
 !_jit_compiled_convolution_op"
_tf_keras_layer
κ
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(axis
	)gamma
*beta
+moving_mean
,moving_variance"
_tf_keras_layer
έ
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias
 5_jit_compiled_convolution_op"
_tf_keras_layer
κ
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
<axis
	=gamma
>beta
?moving_mean
@moving_variance"
_tf_keras_layer
έ
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

Gkernel
Hbias
 I_jit_compiled_convolution_op"
_tf_keras_layer
κ
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance"
_tf_keras_layer
έ
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

[kernel
\bias
 ]_jit_compiled_convolution_op"
_tf_keras_layer
κ
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance"
_tf_keras_layer
έ
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

okernel
pbias
 q_jit_compiled_convolution_op"
_tf_keras_layer
κ
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
xaxis
	ygamma
zbeta
{moving_mean
|moving_variance"
_tf_keras_layer
¨
}	variables
~trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Γ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
Γ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
Γ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
Ό
0
 1
)2
*3
+4
,5
36
47
=8
>9
?10
@11
G12
H13
Q14
R15
S16
T17
[18
\19
e20
f21
g22
h23
o24
p25
y26
z27
{28
|29
30
31
32
33
34
35"
trackable_list_wrapper
μ
0
 1
)2
*3
34
45
=6
>7
G8
H9
Q10
R11
[12
\13
e14
f15
o16
p17
y18
z19
20
21
22
23
24
25"
trackable_list_wrapper
 "
trackable_list_wrapper
Ο
‘non_trainable_variables
’layers
£metrics
 €layer_regularization_losses
₯layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ή
¦trace_0
§trace_1
¨trace_2
©trace_32λ
(__inference_encoder_layer_call_fn_400466
(__inference_encoder_layer_call_fn_401267
(__inference_encoder_layer_call_fn_401348
(__inference_encoder_layer_call_fn_400917ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 z¦trace_0z§trace_1z¨trace_2z©trace_3
Κ
ͺtrace_0
«trace_1
¬trace_2
­trace_32Χ
C__inference_encoder_layer_call_and_return_conditional_losses_401491
C__inference_encoder_layer_call_and_return_conditional_losses_401634
C__inference_encoder_layer_call_and_return_conditional_losses_401010
C__inference_encoder_layer_call_and_return_conditional_losses_401103ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zͺtrace_0z«trace_1z¬trace_2z­trace_3
?BΟ
!__inference__wrapped_model_399852Encoder_Input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
-
?serving_default"
signature_map
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
―non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ο
΄trace_02Π
)__inference_Conv2D_1_layer_call_fn_401643’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z΄trace_0

΅trace_02λ
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_401654’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z΅trace_0
):' 2Conv2D_1/kernel
: 2Conv2D_1/bias
΄2±?
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 0
<
)0
*1
+2
,3"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Άnon_trainable_variables
·layers
Έmetrics
 Ήlayer_regularization_losses
Ίlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
ΐ
»trace_0
Όtrace_12
%__inference_BN_1_layer_call_fn_401667
%__inference_BN_1_layer_call_fn_401680΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 z»trace_0zΌtrace_1
φ
½trace_0
Ύtrace_12»
@__inference_BN_1_layer_call_and_return_conditional_losses_401698
@__inference_BN_1_layer_call_and_return_conditional_losses_401716΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 z½trace_0zΎtrace_1
 "
trackable_list_wrapper
: 2
BN_1/gamma
: 2	BN_1/beta
 :  (2BN_1/moving_mean
$:"  (2BN_1/moving_variance
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ώnon_trainable_variables
ΐlayers
Αmetrics
 Βlayer_regularization_losses
Γlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
ο
Δtrace_02Π
)__inference_Conv2D_2_layer_call_fn_401725’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΔtrace_0

Εtrace_02λ
D__inference_Conv2D_2_layer_call_and_return_conditional_losses_401736’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΕtrace_0
):' @2Conv2D_2/kernel
:@2Conv2D_2/bias
΄2±?
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 0
<
=0
>1
?2
@3"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ζnon_trainable_variables
Ηlayers
Θmetrics
 Ιlayer_regularization_losses
Κlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
ΐ
Λtrace_0
Μtrace_12
%__inference_BN_2_layer_call_fn_401749
%__inference_BN_2_layer_call_fn_401762΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zΛtrace_0zΜtrace_1
φ
Νtrace_0
Ξtrace_12»
@__inference_BN_2_layer_call_and_return_conditional_losses_401780
@__inference_BN_2_layer_call_and_return_conditional_losses_401798΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zΝtrace_0zΞtrace_1
 "
trackable_list_wrapper
:@2
BN_2/gamma
:@2	BN_2/beta
 :@ (2BN_2/moving_mean
$:"@ (2BN_2/moving_variance
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Οnon_trainable_variables
Πlayers
Ρmetrics
 ?layer_regularization_losses
Σlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
ο
Τtrace_02Π
)__inference_Conv2D_3_layer_call_fn_401807’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΤtrace_0

Υtrace_02λ
D__inference_Conv2D_3_layer_call_and_return_conditional_losses_401818’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΥtrace_0
*:(@2Conv2D_3/kernel
:2Conv2D_3/bias
΄2±?
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 0
<
Q0
R1
S2
T3"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Φnon_trainable_variables
Χlayers
Ψmetrics
 Ωlayer_regularization_losses
Ϊlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
ΐ
Ϋtrace_0
άtrace_12
%__inference_BN_3_layer_call_fn_401831
%__inference_BN_3_layer_call_fn_401844΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zΫtrace_0zάtrace_1
φ
έtrace_0
ήtrace_12»
@__inference_BN_3_layer_call_and_return_conditional_losses_401862
@__inference_BN_3_layer_call_and_return_conditional_losses_401880΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zέtrace_0zήtrace_1
 "
trackable_list_wrapper
:2
BN_3/gamma
:2	BN_3/beta
!: (2BN_3/moving_mean
%:# (2BN_3/moving_variance
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ίnon_trainable_variables
ΰlayers
αmetrics
 βlayer_regularization_losses
γlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
ο
δtrace_02Π
)__inference_Conv2D_4_layer_call_fn_401889’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zδtrace_0

εtrace_02λ
D__inference_Conv2D_4_layer_call_and_return_conditional_losses_401900’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zεtrace_0
+:)2Conv2D_4/kernel
:2Conv2D_4/bias
΄2±?
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 0
<
e0
f1
g2
h3"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ζnon_trainable_variables
ηlayers
θmetrics
 ιlayer_regularization_losses
κlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
ΐ
λtrace_0
μtrace_12
%__inference_BN_4_layer_call_fn_401913
%__inference_BN_4_layer_call_fn_401926΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zλtrace_0zμtrace_1
φ
νtrace_0
ξtrace_12»
@__inference_BN_4_layer_call_and_return_conditional_losses_401944
@__inference_BN_4_layer_call_and_return_conditional_losses_401962΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zνtrace_0zξtrace_1
 "
trackable_list_wrapper
:2
BN_4/gamma
:2	BN_4/beta
!: (2BN_4/moving_mean
%:# (2BN_4/moving_variance
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
οnon_trainable_variables
πlayers
ρmetrics
 ςlayer_regularization_losses
σlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
ο
τtrace_02Π
)__inference_Conv2D_5_layer_call_fn_401971’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zτtrace_0

υtrace_02λ
D__inference_Conv2D_5_layer_call_and_return_conditional_losses_401982’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zυtrace_0
+:)2Conv2D_5/kernel
:2Conv2D_5/bias
΄2±?
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 0
<
y0
z1
{2
|3"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
φnon_trainable_variables
χlayers
ψmetrics
 ωlayer_regularization_losses
ϊlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
ΐ
ϋtrace_0
όtrace_12
%__inference_BN_5_layer_call_fn_401995
%__inference_BN_5_layer_call_fn_402008΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zϋtrace_0zόtrace_1
φ
ύtrace_0
ώtrace_12»
@__inference_BN_5_layer_call_and_return_conditional_losses_402026
@__inference_BN_5_layer_call_and_return_conditional_losses_402044΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zύtrace_0zώtrace_1
 "
trackable_list_wrapper
:2
BN_5/gamma
:2	BN_5/beta
!: (2BN_5/moving_mean
%:# (2BN_5/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
?non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ξ
trace_02Ο
(__inference_Flatten_layer_call_fn_402049’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0

trace_02κ
C__inference_Flatten_layer_call_and_return_conditional_losses_402055’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
τ
trace_02Υ
.__inference_Dense_Layer_6_layer_call_fn_402064’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0

trace_02π
I__inference_Dense_Layer_6_layer_call_and_return_conditional_losses_402075’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
(:&
@2Dense_Layer_6/kernel
!:2Dense_Layer_6/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ο
trace_02Π
)__inference_Dense_MU_layer_call_fn_402084’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0

trace_02λ
D__inference_Dense_MU_layer_call_and_return_conditional_losses_402094’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
#:!
2Dense_MU/kernel
:2Dense_MU/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
τ
trace_02Υ
.__inference_Dense_Log_Var_layer_call_fn_402103’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0

trace_02π
I__inference_Dense_Log_Var_layer_call_and_return_conditional_losses_402113’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
(:&
2Dense_Log_Var/kernel
!:2Dense_Log_Var/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
Μ
 trace_0
‘trace_12
%__inference_Code_layer_call_fn_402119
%__inference_Code_layer_call_fn_402125ΐ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 z trace_0z‘trace_1

’trace_0
£trace_12Η
@__inference_Code_layer_call_and_return_conditional_losses_402141
@__inference_Code_layer_call_and_return_conditional_losses_402157ΐ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 z’trace_0z£trace_1
f
+0
,1
?2
@3
S4
T5
g6
h7
{8
|9"
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bώ
(__inference_encoder_layer_call_fn_400466Encoder_Input"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ϊBχ
(__inference_encoder_layer_call_fn_401267inputs"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ϊBχ
(__inference_encoder_layer_call_fn_401348inputs"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Bώ
(__inference_encoder_layer_call_fn_400917Encoder_Input"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
C__inference_encoder_layer_call_and_return_conditional_losses_401491inputs"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
C__inference_encoder_layer_call_and_return_conditional_losses_401634inputs"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
C__inference_encoder_layer_call_and_return_conditional_losses_401010Encoder_Input"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
C__inference_encoder_layer_call_and_return_conditional_losses_401103Encoder_Input"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΡBΞ
$__inference_signature_wrapper_401186Encoder_Input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
έBΪ
)__inference_Conv2D_1_layer_call_fn_401643inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ψBυ
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_401654inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
λBθ
%__inference_BN_1_layer_call_fn_401667inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
λBθ
%__inference_BN_1_layer_call_fn_401680inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
@__inference_BN_1_layer_call_and_return_conditional_losses_401698inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
@__inference_BN_1_layer_call_and_return_conditional_losses_401716inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
έBΪ
)__inference_Conv2D_2_layer_call_fn_401725inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ψBυ
D__inference_Conv2D_2_layer_call_and_return_conditional_losses_401736inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
λBθ
%__inference_BN_2_layer_call_fn_401749inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
λBθ
%__inference_BN_2_layer_call_fn_401762inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
@__inference_BN_2_layer_call_and_return_conditional_losses_401780inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
@__inference_BN_2_layer_call_and_return_conditional_losses_401798inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
έBΪ
)__inference_Conv2D_3_layer_call_fn_401807inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ψBυ
D__inference_Conv2D_3_layer_call_and_return_conditional_losses_401818inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
λBθ
%__inference_BN_3_layer_call_fn_401831inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
λBθ
%__inference_BN_3_layer_call_fn_401844inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
@__inference_BN_3_layer_call_and_return_conditional_losses_401862inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
@__inference_BN_3_layer_call_and_return_conditional_losses_401880inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
έBΪ
)__inference_Conv2D_4_layer_call_fn_401889inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ψBυ
D__inference_Conv2D_4_layer_call_and_return_conditional_losses_401900inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
λBθ
%__inference_BN_4_layer_call_fn_401913inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
λBθ
%__inference_BN_4_layer_call_fn_401926inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
@__inference_BN_4_layer_call_and_return_conditional_losses_401944inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
@__inference_BN_4_layer_call_and_return_conditional_losses_401962inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
έBΪ
)__inference_Conv2D_5_layer_call_fn_401971inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ψBυ
D__inference_Conv2D_5_layer_call_and_return_conditional_losses_401982inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
λBθ
%__inference_BN_5_layer_call_fn_401995inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
λBθ
%__inference_BN_5_layer_call_fn_402008inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
@__inference_BN_5_layer_call_and_return_conditional_losses_402026inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
@__inference_BN_5_layer_call_and_return_conditional_losses_402044inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
άBΩ
(__inference_Flatten_layer_call_fn_402049inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
χBτ
C__inference_Flatten_layer_call_and_return_conditional_losses_402055inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
βBί
.__inference_Dense_Layer_6_layer_call_fn_402064inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ύBϊ
I__inference_Dense_Layer_6_layer_call_and_return_conditional_losses_402075inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
έBΪ
)__inference_Dense_MU_layer_call_fn_402084inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ψBυ
D__inference_Dense_MU_layer_call_and_return_conditional_losses_402094inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
βBί
.__inference_Dense_Log_Var_layer_call_fn_402103inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ύBϊ
I__inference_Dense_Log_Var_layer_call_and_return_conditional_losses_402113inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
%__inference_Code_layer_call_fn_402119inputs/0inputs/1"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
%__inference_Code_layer_call_fn_402125inputs/0inputs/1"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
@__inference_Code_layer_call_and_return_conditional_losses_402141inputs/0inputs/1"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
@__inference_Code_layer_call_and_return_conditional_losses_402157inputs/0inputs/1"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 Ϋ
@__inference_BN_1_layer_call_and_return_conditional_losses_401698)*+,M’J
C’@
:7
inputs+??????????????????????????? 
p 
ͺ "?’<
52
0+??????????????????????????? 
 Ϋ
@__inference_BN_1_layer_call_and_return_conditional_losses_401716)*+,M’J
C’@
:7
inputs+??????????????????????????? 
p
ͺ "?’<
52
0+??????????????????????????? 
 ³
%__inference_BN_1_layer_call_fn_401667)*+,M’J
C’@
:7
inputs+??????????????????????????? 
p 
ͺ "2/+??????????????????????????? ³
%__inference_BN_1_layer_call_fn_401680)*+,M’J
C’@
:7
inputs+??????????????????????????? 
p
ͺ "2/+??????????????????????????? Ϋ
@__inference_BN_2_layer_call_and_return_conditional_losses_401780=>?@M’J
C’@
:7
inputs+???????????????????????????@
p 
ͺ "?’<
52
0+???????????????????????????@
 Ϋ
@__inference_BN_2_layer_call_and_return_conditional_losses_401798=>?@M’J
C’@
:7
inputs+???????????????????????????@
p
ͺ "?’<
52
0+???????????????????????????@
 ³
%__inference_BN_2_layer_call_fn_401749=>?@M’J
C’@
:7
inputs+???????????????????????????@
p 
ͺ "2/+???????????????????????????@³
%__inference_BN_2_layer_call_fn_401762=>?@M’J
C’@
:7
inputs+???????????????????????????@
p
ͺ "2/+???????????????????????????@έ
@__inference_BN_3_layer_call_and_return_conditional_losses_401862QRSTN’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "@’=
63
0,???????????????????????????
 έ
@__inference_BN_3_layer_call_and_return_conditional_losses_401880QRSTN’K
D’A
;8
inputs,???????????????????????????
p
ͺ "@’=
63
0,???????????????????????????
 ΅
%__inference_BN_3_layer_call_fn_401831QRSTN’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "30,???????????????????????????΅
%__inference_BN_3_layer_call_fn_401844QRSTN’K
D’A
;8
inputs,???????????????????????????
p
ͺ "30,???????????????????????????έ
@__inference_BN_4_layer_call_and_return_conditional_losses_401944efghN’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "@’=
63
0,???????????????????????????
 έ
@__inference_BN_4_layer_call_and_return_conditional_losses_401962efghN’K
D’A
;8
inputs,???????????????????????????
p
ͺ "@’=
63
0,???????????????????????????
 ΅
%__inference_BN_4_layer_call_fn_401913efghN’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "30,???????????????????????????΅
%__inference_BN_4_layer_call_fn_401926efghN’K
D’A
;8
inputs,???????????????????????????
p
ͺ "30,???????????????????????????έ
@__inference_BN_5_layer_call_and_return_conditional_losses_402026yz{|N’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "@’=
63
0,???????????????????????????
 έ
@__inference_BN_5_layer_call_and_return_conditional_losses_402044yz{|N’K
D’A
;8
inputs,???????????????????????????
p
ͺ "@’=
63
0,???????????????????????????
 ΅
%__inference_BN_5_layer_call_fn_401995yz{|N’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "30,???????????????????????????΅
%__inference_BN_5_layer_call_fn_402008yz{|N’K
D’A
;8
inputs,???????????????????????????
p
ͺ "30,???????????????????????????Σ
@__inference_Code_layer_call_and_return_conditional_losses_402141d’a
Z’W
MJ
# 
inputs/0?????????
# 
inputs/1?????????

 
p 
ͺ "&’#

0?????????
 Σ
@__inference_Code_layer_call_and_return_conditional_losses_402157d’a
Z’W
MJ
# 
inputs/0?????????
# 
inputs/1?????????

 
p
ͺ "&’#

0?????????
 «
%__inference_Code_layer_call_fn_402119d’a
Z’W
MJ
# 
inputs/0?????????
# 
inputs/1?????????

 
p 
ͺ "?????????«
%__inference_Code_layer_call_fn_402125d’a
Z’W
MJ
# 
inputs/0?????????
# 
inputs/1?????????

 
p
ͺ "?????????Ά
D__inference_Conv2D_1_layer_call_and_return_conditional_losses_401654n 9’6
/’,
*'
inputs?????????
ͺ "-’*
# 
0?????????@@ 
 
)__inference_Conv2D_1_layer_call_fn_401643a 9’6
/’,
*'
inputs?????????
ͺ " ?????????@@ ΄
D__inference_Conv2D_2_layer_call_and_return_conditional_losses_401736l347’4
-’*
(%
inputs?????????@@ 
ͺ "-’*
# 
0?????????  @
 
)__inference_Conv2D_2_layer_call_fn_401725_347’4
-’*
(%
inputs?????????@@ 
ͺ " ?????????  @΅
D__inference_Conv2D_3_layer_call_and_return_conditional_losses_401818mGH7’4
-’*
(%
inputs?????????  @
ͺ ".’+
$!
0?????????
 
)__inference_Conv2D_3_layer_call_fn_401807`GH7’4
-’*
(%
inputs?????????  @
ͺ "!?????????Ά
D__inference_Conv2D_4_layer_call_and_return_conditional_losses_401900n[\8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
)__inference_Conv2D_4_layer_call_fn_401889a[\8’5
.’+
)&
inputs?????????
ͺ "!?????????Ά
D__inference_Conv2D_5_layer_call_and_return_conditional_losses_401982nop8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
)__inference_Conv2D_5_layer_call_fn_401971aop8’5
.’+
)&
inputs?????????
ͺ "!?????????­
I__inference_Dense_Layer_6_layer_call_and_return_conditional_losses_402075`0’-
&’#
!
inputs?????????@
ͺ "&’#

0?????????
 
.__inference_Dense_Layer_6_layer_call_fn_402064S0’-
&’#
!
inputs?????????@
ͺ "?????????­
I__inference_Dense_Log_Var_layer_call_and_return_conditional_losses_402113`0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 
.__inference_Dense_Log_Var_layer_call_fn_402103S0’-
&’#
!
inputs?????????
ͺ "?????????¨
D__inference_Dense_MU_layer_call_and_return_conditional_losses_402094`0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 
)__inference_Dense_MU_layer_call_fn_402084S0’-
&’#
!
inputs?????????
ͺ "?????????©
C__inference_Flatten_layer_call_and_return_conditional_losses_402055b8’5
.’+
)&
inputs?????????
ͺ "&’#

0?????????@
 
(__inference_Flatten_layer_call_fn_402049U8’5
.’+
)&
inputs?????????
ͺ "?????????@°
!__inference__wrapped_model_399852* )*+,34=>?@GHQRST[\efghopyz{|@’=
6’3
1.
Encoder_Input?????????
ͺ "ͺ
'
Code
Code?????????
9
Dense_Log_Var(%
Dense_Log_Var?????????
/
Dense_MU# 
Dense_MU?????????­
C__inference_encoder_layer_call_and_return_conditional_losses_401010ε* )*+,34=>?@GHQRST[\efghopyz{|H’E
>’;
1.
Encoder_Input?????????
p 

 
ͺ "m’j
c`

0/0?????????

0/1?????????

0/2?????????
 ­
C__inference_encoder_layer_call_and_return_conditional_losses_401103ε* )*+,34=>?@GHQRST[\efghopyz{|H’E
>’;
1.
Encoder_Input?????????
p

 
ͺ "m’j
c`

0/0?????????

0/1?????????

0/2?????????
 ¦
C__inference_encoder_layer_call_and_return_conditional_losses_401491ή* )*+,34=>?@GHQRST[\efghopyz{|A’>
7’4
*'
inputs?????????
p 

 
ͺ "m’j
c`

0/0?????????

0/1?????????

0/2?????????
 ¦
C__inference_encoder_layer_call_and_return_conditional_losses_401634ή* )*+,34=>?@GHQRST[\efghopyz{|A’>
7’4
*'
inputs?????????
p

 
ͺ "m’j
c`

0/0?????????

0/1?????????

0/2?????????
 
(__inference_encoder_layer_call_fn_400466Υ* )*+,34=>?@GHQRST[\efghopyz{|H’E
>’;
1.
Encoder_Input?????????
p 

 
ͺ "]Z

0?????????

1?????????

2?????????
(__inference_encoder_layer_call_fn_400917Υ* )*+,34=>?@GHQRST[\efghopyz{|H’E
>’;
1.
Encoder_Input?????????
p

 
ͺ "]Z

0?????????

1?????????

2?????????ϋ
(__inference_encoder_layer_call_fn_401267Ξ* )*+,34=>?@GHQRST[\efghopyz{|A’>
7’4
*'
inputs?????????
p 

 
ͺ "]Z

0?????????

1?????????

2?????????ϋ
(__inference_encoder_layer_call_fn_401348Ξ* )*+,34=>?@GHQRST[\efghopyz{|A’>
7’4
*'
inputs?????????
p

 
ͺ "]Z

0?????????

1?????????

2?????????Δ
$__inference_signature_wrapper_401186* )*+,34=>?@GHQRST[\efghopyz{|Q’N
’ 
GͺD
B
Encoder_Input1.
Encoder_Input?????????"ͺ
'
Code
Code?????????
9
Dense_Log_Var(%
Dense_Log_Var?????????
/
Dense_MU# 
Dense_MU?????????