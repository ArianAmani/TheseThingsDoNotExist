??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
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
?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
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
?
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
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
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
delete_old_dirsbool(?
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.22v2.9.1-132-g18960c44ad38??
?
Conv2DTranspose_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameConv2DTranspose_13/bias

+Conv2DTranspose_13/bias/Read/ReadVariableOpReadVariableOpConv2DTranspose_13/bias*
_output_shapes
:*
dtype0
?
Conv2DTranspose_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameConv2DTranspose_13/kernel
?
-Conv2DTranspose_13/kernel/Read/ReadVariableOpReadVariableOpConv2DTranspose_13/kernel*&
_output_shapes
: *
dtype0
?
BN_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameBN_12/moving_variance
{
)BN_12/moving_variance/Read/ReadVariableOpReadVariableOpBN_12/moving_variance*
_output_shapes
: *
dtype0
z
BN_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameBN_12/moving_mean
s
%BN_12/moving_mean/Read/ReadVariableOpReadVariableOpBN_12/moving_mean*
_output_shapes
: *
dtype0
l

BN_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
BN_12/beta
e
BN_12/beta/Read/ReadVariableOpReadVariableOp
BN_12/beta*
_output_shapes
: *
dtype0
n
BN_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameBN_12/gamma
g
BN_12/gamma/Read/ReadVariableOpReadVariableOpBN_12/gamma*
_output_shapes
: *
dtype0
?
Conv2DTranspose_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameConv2DTranspose_12/bias

+Conv2DTranspose_12/bias/Read/ReadVariableOpReadVariableOpConv2DTranspose_12/bias*
_output_shapes
: *
dtype0
?
Conv2DTranspose_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameConv2DTranspose_12/kernel
?
-Conv2DTranspose_12/kernel/Read/ReadVariableOpReadVariableOpConv2DTranspose_12/kernel*&
_output_shapes
: @*
dtype0
?
BN_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameBN_11/moving_variance
{
)BN_11/moving_variance/Read/ReadVariableOpReadVariableOpBN_11/moving_variance*
_output_shapes
:@*
dtype0
z
BN_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameBN_11/moving_mean
s
%BN_11/moving_mean/Read/ReadVariableOpReadVariableOpBN_11/moving_mean*
_output_shapes
:@*
dtype0
l

BN_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
BN_11/beta
e
BN_11/beta/Read/ReadVariableOpReadVariableOp
BN_11/beta*
_output_shapes
:@*
dtype0
n
BN_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameBN_11/gamma
g
BN_11/gamma/Read/ReadVariableOpReadVariableOpBN_11/gamma*
_output_shapes
:@*
dtype0
?
Conv2DTranspose_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameConv2DTranspose_11/bias

+Conv2DTranspose_11/bias/Read/ReadVariableOpReadVariableOpConv2DTranspose_11/bias*
_output_shapes
:@*
dtype0
?
Conv2DTranspose_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?**
shared_nameConv2DTranspose_11/kernel
?
-Conv2DTranspose_11/kernel/Read/ReadVariableOpReadVariableOpConv2DTranspose_11/kernel*'
_output_shapes
:@?*
dtype0
?
BN_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameBN_10/moving_variance
|
)BN_10/moving_variance/Read/ReadVariableOpReadVariableOpBN_10/moving_variance*
_output_shapes	
:?*
dtype0
{
BN_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameBN_10/moving_mean
t
%BN_10/moving_mean/Read/ReadVariableOpReadVariableOpBN_10/moving_mean*
_output_shapes	
:?*
dtype0
m

BN_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
BN_10/beta
f
BN_10/beta/Read/ReadVariableOpReadVariableOp
BN_10/beta*
_output_shapes	
:?*
dtype0
o
BN_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameBN_10/gamma
h
BN_10/gamma/Read/ReadVariableOpReadVariableOpBN_10/gamma*
_output_shapes	
:?*
dtype0
?
Conv2DTranspose_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameConv2DTranspose_10/bias
?
+Conv2DTranspose_10/bias/Read/ReadVariableOpReadVariableOpConv2DTranspose_10/bias*
_output_shapes	
:?*
dtype0
?
Conv2DTranspose_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??**
shared_nameConv2DTranspose_10/kernel
?
-Conv2DTranspose_10/kernel/Read/ReadVariableOpReadVariableOpConv2DTranspose_10/kernel*(
_output_shapes
:??*
dtype0
?
BN_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameBN_9/moving_variance
z
(BN_9/moving_variance/Read/ReadVariableOpReadVariableOpBN_9/moving_variance*
_output_shapes	
:?*
dtype0
y
BN_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameBN_9/moving_mean
r
$BN_9/moving_mean/Read/ReadVariableOpReadVariableOpBN_9/moving_mean*
_output_shapes	
:?*
dtype0
k
	BN_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	BN_9/beta
d
BN_9/beta/Read/ReadVariableOpReadVariableOp	BN_9/beta*
_output_shapes	
:?*
dtype0
m

BN_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
BN_9/gamma
f
BN_9/gamma/Read/ReadVariableOpReadVariableOp
BN_9/gamma*
_output_shapes	
:?*
dtype0
?
Conv2DTranspose_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameConv2DTranspose_9/bias
~
*Conv2DTranspose_9/bias/Read/ReadVariableOpReadVariableOpConv2DTranspose_9/bias*
_output_shapes	
:?*
dtype0
?
Conv2DTranspose_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*)
shared_nameConv2DTranspose_9/kernel
?
,Conv2DTranspose_9/kernel/Read/ReadVariableOpReadVariableOpConv2DTranspose_9/kernel*(
_output_shapes
:??*
dtype0
}
Dense_Layer_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*#
shared_nameDense_Layer_8/bias
v
&Dense_Layer_8/bias/Read/ReadVariableOpReadVariableOpDense_Layer_8/bias*
_output_shapes	
:?@*
dtype0
?
Dense_Layer_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*%
shared_nameDense_Layer_8/kernel

(Dense_Layer_8/kernel/Read/ReadVariableOpReadVariableOpDense_Layer_8/kernel* 
_output_shapes
:
??@*
dtype0
}
Dense_Layer_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_nameDense_Layer_7/bias
v
&Dense_Layer_7/bias/Read/ReadVariableOpReadVariableOpDense_Layer_7/bias*
_output_shapes	
:?*
dtype0
?
Dense_Layer_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*%
shared_nameDense_Layer_7/kernel

(Dense_Layer_7/kernel/Read/ReadVariableOpReadVariableOpDense_Layer_7/kernel* 
_output_shapes
:
??*
dtype0

NoOpNoOp
?_
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?^
value?^B?^ B?^
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
?
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses* 
?
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op*
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;axis
	<gamma
=beta
>moving_mean
?moving_variance*
?
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias
 H_jit_compiled_convolution_op*
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance*
?
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

Zkernel
[bias
 \_jit_compiled_convolution_op*
?
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses
caxis
	dgamma
ebeta
fmoving_mean
gmoving_variance*
?
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses

nkernel
obias
 p_jit_compiled_convolution_op*
?
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance*
?
|	variables
}trainable_variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op*
?
0
1
$2
%3
24
35
<6
=7
>8
?9
F10
G11
P12
Q13
R14
S15
Z16
[17
d18
e19
f20
g21
n22
o23
x24
y25
z26
{27
?28
?29*
?
0
1
$2
%3
24
35
<6
=7
F8
G9
P10
Q11
Z12
[13
d14
e15
n16
o17
x18
y19
?20
?21*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 

?serving_default* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
d^
VARIABLE_VALUEDense_Layer_7/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEDense_Layer_7/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
d^
VARIABLE_VALUEDense_Layer_8/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEDense_Layer_8/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

20
31*

20
31*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
hb
VARIABLE_VALUEConv2DTranspose_9/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEConv2DTranspose_9/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
<0
=1
>2
?3*

<0
=1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
YS
VARIABLE_VALUE
BN_9/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	BN_9/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEBN_9/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEBN_9/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

F0
G1*

F0
G1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
ic
VARIABLE_VALUEConv2DTranspose_10/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEConv2DTranspose_10/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
P0
Q1
R2
S3*

P0
Q1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ZT
VARIABLE_VALUEBN_10/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
BN_10/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEBN_10/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEBN_10/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

Z0
[1*

Z0
[1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
ic
VARIABLE_VALUEConv2DTranspose_11/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEConv2DTranspose_11/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
d0
e1
f2
g3*

d0
e1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ZT
VARIABLE_VALUEBN_11/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
BN_11/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEBN_11/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEBN_11/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

n0
o1*

n0
o1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
ic
VARIABLE_VALUEConv2DTranspose_12/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEConv2DTranspose_12/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
x0
y1
z2
{3*

x0
y1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ZT
VARIABLE_VALUEBN_12/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
BN_12/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEBN_12/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEBN_12/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
jd
VARIABLE_VALUEConv2DTranspose_13/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEConv2DTranspose_13/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
>0
?1
R2
S3
f4
g5
z6
{7*
b
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
12*
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

>0
?1*
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
R0
S1*
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
f0
g1*
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
z0
{1*
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
?
serving_default_Decoder_InputPlaceholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_Decoder_InputDense_Layer_7/kernelDense_Layer_7/biasDense_Layer_8/kernelDense_Layer_8/biasConv2DTranspose_9/kernelConv2DTranspose_9/bias
BN_9/gamma	BN_9/betaBN_9/moving_meanBN_9/moving_varianceConv2DTranspose_10/kernelConv2DTranspose_10/biasBN_10/gamma
BN_10/betaBN_10/moving_meanBN_10/moving_varianceConv2DTranspose_11/kernelConv2DTranspose_11/biasBN_11/gamma
BN_11/betaBN_11/moving_meanBN_11/moving_varianceConv2DTranspose_12/kernelConv2DTranspose_12/biasBN_12/gamma
BN_12/betaBN_12/moving_meanBN_12/moving_varianceConv2DTranspose_13/kernelConv2DTranspose_13/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_403931
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(Dense_Layer_7/kernel/Read/ReadVariableOp&Dense_Layer_7/bias/Read/ReadVariableOp(Dense_Layer_8/kernel/Read/ReadVariableOp&Dense_Layer_8/bias/Read/ReadVariableOp,Conv2DTranspose_9/kernel/Read/ReadVariableOp*Conv2DTranspose_9/bias/Read/ReadVariableOpBN_9/gamma/Read/ReadVariableOpBN_9/beta/Read/ReadVariableOp$BN_9/moving_mean/Read/ReadVariableOp(BN_9/moving_variance/Read/ReadVariableOp-Conv2DTranspose_10/kernel/Read/ReadVariableOp+Conv2DTranspose_10/bias/Read/ReadVariableOpBN_10/gamma/Read/ReadVariableOpBN_10/beta/Read/ReadVariableOp%BN_10/moving_mean/Read/ReadVariableOp)BN_10/moving_variance/Read/ReadVariableOp-Conv2DTranspose_11/kernel/Read/ReadVariableOp+Conv2DTranspose_11/bias/Read/ReadVariableOpBN_11/gamma/Read/ReadVariableOpBN_11/beta/Read/ReadVariableOp%BN_11/moving_mean/Read/ReadVariableOp)BN_11/moving_variance/Read/ReadVariableOp-Conv2DTranspose_12/kernel/Read/ReadVariableOp+Conv2DTranspose_12/bias/Read/ReadVariableOpBN_12/gamma/Read/ReadVariableOpBN_12/beta/Read/ReadVariableOp%BN_12/moving_mean/Read/ReadVariableOp)BN_12/moving_variance/Read/ReadVariableOp-Conv2DTranspose_13/kernel/Read/ReadVariableOp+Conv2DTranspose_13/bias/Read/ReadVariableOpConst*+
Tin$
"2 *
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
GPU2*0J 8? *(
f#R!
__inference__traced_save_405064
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameDense_Layer_7/kernelDense_Layer_7/biasDense_Layer_8/kernelDense_Layer_8/biasConv2DTranspose_9/kernelConv2DTranspose_9/bias
BN_9/gamma	BN_9/betaBN_9/moving_meanBN_9/moving_varianceConv2DTranspose_10/kernelConv2DTranspose_10/biasBN_10/gamma
BN_10/betaBN_10/moving_meanBN_10/moving_varianceConv2DTranspose_11/kernelConv2DTranspose_11/biasBN_11/gamma
BN_11/betaBN_11/moving_meanBN_11/moving_varianceConv2DTranspose_12/kernelConv2DTranspose_12/biasBN_12/gamma
BN_12/betaBN_12/moving_meanBN_12/moving_varianceConv2DTranspose_13/kernelConv2DTranspose_13/bias**
Tin#
!2*
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
GPU2*0J 8? *+
f&R$
"__inference__traced_restore_405164??
?
?
(__inference_decoder_layer_call_fn_404061

inputs
unknown:
??
	unknown_0:	?
	unknown_1:
??@
	unknown_2:	?@%
	unknown_3:??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?%

unknown_15:@?

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@$

unknown_21: @

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: $

unknown_27: 

unknown_28:
identity??StatefulPartitionedCall?
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*8
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_403584y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_BN_9_layer_call_and_return_conditional_losses_404593

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_403931
decoder_input
unknown:
??
	unknown_0:	?
	unknown_1:
??@
	unknown_2:	?@%
	unknown_3:??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?%

unknown_15:@?

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@$

unknown_21: @

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: $

unknown_27: 

unknown_28:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldecoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_402752y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameDecoder_Input
?
?
&__inference_BN_12_layer_call_fn_404859

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
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
GPU2*0J 8? *J
fERC
A__inference_BN_12_layer_call_and_return_conditional_losses_403146?
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
?C
?
C__inference_decoder_layer_call_and_return_conditional_losses_403584

inputs(
dense_layer_7_403511:
??#
dense_layer_7_403513:	?(
dense_layer_8_403516:
??@#
dense_layer_8_403518:	?@4
conv2dtranspose_9_403522:??'
conv2dtranspose_9_403524:	?
bn_9_403527:	?
bn_9_403529:	?
bn_9_403531:	?
bn_9_403533:	?5
conv2dtranspose_10_403536:??(
conv2dtranspose_10_403538:	?
bn_10_403541:	?
bn_10_403543:	?
bn_10_403545:	?
bn_10_403547:	?4
conv2dtranspose_11_403550:@?'
conv2dtranspose_11_403552:@
bn_11_403555:@
bn_11_403557:@
bn_11_403559:@
bn_11_403561:@3
conv2dtranspose_12_403564: @'
conv2dtranspose_12_403566: 
bn_12_403569: 
bn_12_403571: 
bn_12_403573: 
bn_12_403575: 3
conv2dtranspose_13_403578: '
conv2dtranspose_13_403580:
identity??BN_10/StatefulPartitionedCall?BN_11/StatefulPartitionedCall?BN_12/StatefulPartitionedCall?BN_9/StatefulPartitionedCall?*Conv2DTranspose_10/StatefulPartitionedCall?*Conv2DTranspose_11/StatefulPartitionedCall?*Conv2DTranspose_12/StatefulPartitionedCall?*Conv2DTranspose_13/StatefulPartitionedCall?)Conv2DTranspose_9/StatefulPartitionedCall?%Dense_Layer_7/StatefulPartitionedCall?%Dense_Layer_8/StatefulPartitionedCall?
%Dense_Layer_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_layer_7_403511dense_layer_7_403513*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Dense_Layer_7_layer_call_and_return_conditional_losses_403251?
%Dense_Layer_8/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_7/StatefulPartitionedCall:output:0dense_layer_8_403516dense_layer_8_403518*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Dense_Layer_8_layer_call_and_return_conditional_losses_403268?
Reshape_Layer/PartitionedCallPartitionedCall.Dense_Layer_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Reshape_Layer_layer_call_and_return_conditional_losses_403288?
)Conv2DTranspose_9/StatefulPartitionedCallStatefulPartitionedCall&Reshape_Layer/PartitionedCall:output:0conv2dtranspose_9_403522conv2dtranspose_9_403524*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_Conv2DTranspose_9_layer_call_and_return_conditional_losses_402790?
BN_9/StatefulPartitionedCallStatefulPartitionedCall2Conv2DTranspose_9/StatefulPartitionedCall:output:0bn_9_403527bn_9_403529bn_9_403531bn_9_403533*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_BN_9_layer_call_and_return_conditional_losses_402850?
*Conv2DTranspose_10/StatefulPartitionedCallStatefulPartitionedCall%BN_9/StatefulPartitionedCall:output:0conv2dtranspose_10_403536conv2dtranspose_10_403538*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_10_layer_call_and_return_conditional_losses_402899?
BN_10/StatefulPartitionedCallStatefulPartitionedCall3Conv2DTranspose_10/StatefulPartitionedCall:output:0bn_10_403541bn_10_403543bn_10_403545bn_10_403547*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_BN_10_layer_call_and_return_conditional_losses_402959?
*Conv2DTranspose_11/StatefulPartitionedCallStatefulPartitionedCall&BN_10/StatefulPartitionedCall:output:0conv2dtranspose_11_403550conv2dtranspose_11_403552*
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
GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_11_layer_call_and_return_conditional_losses_403008?
BN_11/StatefulPartitionedCallStatefulPartitionedCall3Conv2DTranspose_11/StatefulPartitionedCall:output:0bn_11_403555bn_11_403557bn_11_403559bn_11_403561*
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
GPU2*0J 8? *J
fERC
A__inference_BN_11_layer_call_and_return_conditional_losses_403068?
*Conv2DTranspose_12/StatefulPartitionedCallStatefulPartitionedCall&BN_11/StatefulPartitionedCall:output:0conv2dtranspose_12_403564conv2dtranspose_12_403566*
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
GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_12_layer_call_and_return_conditional_losses_403117?
BN_12/StatefulPartitionedCallStatefulPartitionedCall3Conv2DTranspose_12/StatefulPartitionedCall:output:0bn_12_403569bn_12_403571bn_12_403573bn_12_403575*
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
GPU2*0J 8? *J
fERC
A__inference_BN_12_layer_call_and_return_conditional_losses_403177?
*Conv2DTranspose_13/StatefulPartitionedCallStatefulPartitionedCall&BN_12/StatefulPartitionedCall:output:0conv2dtranspose_13_403578conv2dtranspose_13_403580*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_13_layer_call_and_return_conditional_losses_403226?
IdentityIdentity3Conv2DTranspose_13/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp^BN_10/StatefulPartitionedCall^BN_11/StatefulPartitionedCall^BN_12/StatefulPartitionedCall^BN_9/StatefulPartitionedCall+^Conv2DTranspose_10/StatefulPartitionedCall+^Conv2DTranspose_11/StatefulPartitionedCall+^Conv2DTranspose_12/StatefulPartitionedCall+^Conv2DTranspose_13/StatefulPartitionedCall*^Conv2DTranspose_9/StatefulPartitionedCall&^Dense_Layer_7/StatefulPartitionedCall&^Dense_Layer_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
BN_10/StatefulPartitionedCallBN_10/StatefulPartitionedCall2>
BN_11/StatefulPartitionedCallBN_11/StatefulPartitionedCall2>
BN_12/StatefulPartitionedCallBN_12/StatefulPartitionedCall2<
BN_9/StatefulPartitionedCallBN_9/StatefulPartitionedCall2X
*Conv2DTranspose_10/StatefulPartitionedCall*Conv2DTranspose_10/StatefulPartitionedCall2X
*Conv2DTranspose_11/StatefulPartitionedCall*Conv2DTranspose_11/StatefulPartitionedCall2X
*Conv2DTranspose_12/StatefulPartitionedCall*Conv2DTranspose_12/StatefulPartitionedCall2X
*Conv2DTranspose_13/StatefulPartitionedCall*Conv2DTranspose_13/StatefulPartitionedCall2V
)Conv2DTranspose_9/StatefulPartitionedCall)Conv2DTranspose_9/StatefulPartitionedCall2N
%Dense_Layer_7/StatefulPartitionedCall%Dense_Layer_7/StatefulPartitionedCall2N
%Dense_Layer_8/StatefulPartitionedCall%Dense_Layer_8/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
3__inference_Conv2DTranspose_13_layer_call_fn_404917

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_13_layer_call_and_return_conditional_losses_403226?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?!
?
N__inference_Conv2DTranspose_10_layer_call_and_return_conditional_losses_402899

inputsD
(conv2d_transpose_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????k
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_BN_11_layer_call_and_return_conditional_losses_404803

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
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
?

?
I__inference_Dense_Layer_7_layer_call_and_return_conditional_losses_404449

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_Dense_Layer_8_layer_call_fn_404458

inputs
unknown:
??@
	unknown_0:	?@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Dense_Layer_8_layer_call_and_return_conditional_losses_403268p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_BN_12_layer_call_fn_404872

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
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
GPU2*0J 8? *J
fERC
A__inference_BN_12_layer_call_and_return_conditional_losses_403177?
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
?
?
A__inference_BN_10_layer_call_and_return_conditional_losses_402959

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_BN_12_layer_call_and_return_conditional_losses_403146

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
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
?C
?
C__inference_decoder_layer_call_and_return_conditional_losses_403864
decoder_input(
dense_layer_7_403791:
??#
dense_layer_7_403793:	?(
dense_layer_8_403796:
??@#
dense_layer_8_403798:	?@4
conv2dtranspose_9_403802:??'
conv2dtranspose_9_403804:	?
bn_9_403807:	?
bn_9_403809:	?
bn_9_403811:	?
bn_9_403813:	?5
conv2dtranspose_10_403816:??(
conv2dtranspose_10_403818:	?
bn_10_403821:	?
bn_10_403823:	?
bn_10_403825:	?
bn_10_403827:	?4
conv2dtranspose_11_403830:@?'
conv2dtranspose_11_403832:@
bn_11_403835:@
bn_11_403837:@
bn_11_403839:@
bn_11_403841:@3
conv2dtranspose_12_403844: @'
conv2dtranspose_12_403846: 
bn_12_403849: 
bn_12_403851: 
bn_12_403853: 
bn_12_403855: 3
conv2dtranspose_13_403858: '
conv2dtranspose_13_403860:
identity??BN_10/StatefulPartitionedCall?BN_11/StatefulPartitionedCall?BN_12/StatefulPartitionedCall?BN_9/StatefulPartitionedCall?*Conv2DTranspose_10/StatefulPartitionedCall?*Conv2DTranspose_11/StatefulPartitionedCall?*Conv2DTranspose_12/StatefulPartitionedCall?*Conv2DTranspose_13/StatefulPartitionedCall?)Conv2DTranspose_9/StatefulPartitionedCall?%Dense_Layer_7/StatefulPartitionedCall?%Dense_Layer_8/StatefulPartitionedCall?
%Dense_Layer_7/StatefulPartitionedCallStatefulPartitionedCalldecoder_inputdense_layer_7_403791dense_layer_7_403793*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Dense_Layer_7_layer_call_and_return_conditional_losses_403251?
%Dense_Layer_8/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_7/StatefulPartitionedCall:output:0dense_layer_8_403796dense_layer_8_403798*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Dense_Layer_8_layer_call_and_return_conditional_losses_403268?
Reshape_Layer/PartitionedCallPartitionedCall.Dense_Layer_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Reshape_Layer_layer_call_and_return_conditional_losses_403288?
)Conv2DTranspose_9/StatefulPartitionedCallStatefulPartitionedCall&Reshape_Layer/PartitionedCall:output:0conv2dtranspose_9_403802conv2dtranspose_9_403804*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_Conv2DTranspose_9_layer_call_and_return_conditional_losses_402790?
BN_9/StatefulPartitionedCallStatefulPartitionedCall2Conv2DTranspose_9/StatefulPartitionedCall:output:0bn_9_403807bn_9_403809bn_9_403811bn_9_403813*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_BN_9_layer_call_and_return_conditional_losses_402850?
*Conv2DTranspose_10/StatefulPartitionedCallStatefulPartitionedCall%BN_9/StatefulPartitionedCall:output:0conv2dtranspose_10_403816conv2dtranspose_10_403818*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_10_layer_call_and_return_conditional_losses_402899?
BN_10/StatefulPartitionedCallStatefulPartitionedCall3Conv2DTranspose_10/StatefulPartitionedCall:output:0bn_10_403821bn_10_403823bn_10_403825bn_10_403827*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_BN_10_layer_call_and_return_conditional_losses_402959?
*Conv2DTranspose_11/StatefulPartitionedCallStatefulPartitionedCall&BN_10/StatefulPartitionedCall:output:0conv2dtranspose_11_403830conv2dtranspose_11_403832*
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
GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_11_layer_call_and_return_conditional_losses_403008?
BN_11/StatefulPartitionedCallStatefulPartitionedCall3Conv2DTranspose_11/StatefulPartitionedCall:output:0bn_11_403835bn_11_403837bn_11_403839bn_11_403841*
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
GPU2*0J 8? *J
fERC
A__inference_BN_11_layer_call_and_return_conditional_losses_403068?
*Conv2DTranspose_12/StatefulPartitionedCallStatefulPartitionedCall&BN_11/StatefulPartitionedCall:output:0conv2dtranspose_12_403844conv2dtranspose_12_403846*
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
GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_12_layer_call_and_return_conditional_losses_403117?
BN_12/StatefulPartitionedCallStatefulPartitionedCall3Conv2DTranspose_12/StatefulPartitionedCall:output:0bn_12_403849bn_12_403851bn_12_403853bn_12_403855*
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
GPU2*0J 8? *J
fERC
A__inference_BN_12_layer_call_and_return_conditional_losses_403177?
*Conv2DTranspose_13/StatefulPartitionedCallStatefulPartitionedCall&BN_12/StatefulPartitionedCall:output:0conv2dtranspose_13_403858conv2dtranspose_13_403860*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_13_layer_call_and_return_conditional_losses_403226?
IdentityIdentity3Conv2DTranspose_13/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp^BN_10/StatefulPartitionedCall^BN_11/StatefulPartitionedCall^BN_12/StatefulPartitionedCall^BN_9/StatefulPartitionedCall+^Conv2DTranspose_10/StatefulPartitionedCall+^Conv2DTranspose_11/StatefulPartitionedCall+^Conv2DTranspose_12/StatefulPartitionedCall+^Conv2DTranspose_13/StatefulPartitionedCall*^Conv2DTranspose_9/StatefulPartitionedCall&^Dense_Layer_7/StatefulPartitionedCall&^Dense_Layer_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
BN_10/StatefulPartitionedCallBN_10/StatefulPartitionedCall2>
BN_11/StatefulPartitionedCallBN_11/StatefulPartitionedCall2>
BN_12/StatefulPartitionedCallBN_12/StatefulPartitionedCall2<
BN_9/StatefulPartitionedCallBN_9/StatefulPartitionedCall2X
*Conv2DTranspose_10/StatefulPartitionedCall*Conv2DTranspose_10/StatefulPartitionedCall2X
*Conv2DTranspose_11/StatefulPartitionedCall*Conv2DTranspose_11/StatefulPartitionedCall2X
*Conv2DTranspose_12/StatefulPartitionedCall*Conv2DTranspose_12/StatefulPartitionedCall2X
*Conv2DTranspose_13/StatefulPartitionedCall*Conv2DTranspose_13/StatefulPartitionedCall2V
)Conv2DTranspose_9/StatefulPartitionedCall)Conv2DTranspose_9/StatefulPartitionedCall2N
%Dense_Layer_7/StatefulPartitionedCall%Dense_Layer_7/StatefulPartitionedCall2N
%Dense_Layer_8/StatefulPartitionedCall%Dense_Layer_8/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameDecoder_Input
?
?
%__inference_BN_9_layer_call_fn_404557

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_BN_9_layer_call_and_return_conditional_losses_402850?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?!
?
N__inference_Conv2DTranspose_12_layer_call_and_return_conditional_losses_404846

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
e
I__inference_Reshape_Layer_layer_call_and_return_conditional_losses_403288

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :R
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????a
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????@:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?A
?
__inference__traced_save_405064
file_prefix3
/savev2_dense_layer_7_kernel_read_readvariableop1
-savev2_dense_layer_7_bias_read_readvariableop3
/savev2_dense_layer_8_kernel_read_readvariableop1
-savev2_dense_layer_8_bias_read_readvariableop7
3savev2_conv2dtranspose_9_kernel_read_readvariableop5
1savev2_conv2dtranspose_9_bias_read_readvariableop)
%savev2_bn_9_gamma_read_readvariableop(
$savev2_bn_9_beta_read_readvariableop/
+savev2_bn_9_moving_mean_read_readvariableop3
/savev2_bn_9_moving_variance_read_readvariableop8
4savev2_conv2dtranspose_10_kernel_read_readvariableop6
2savev2_conv2dtranspose_10_bias_read_readvariableop*
&savev2_bn_10_gamma_read_readvariableop)
%savev2_bn_10_beta_read_readvariableop0
,savev2_bn_10_moving_mean_read_readvariableop4
0savev2_bn_10_moving_variance_read_readvariableop8
4savev2_conv2dtranspose_11_kernel_read_readvariableop6
2savev2_conv2dtranspose_11_bias_read_readvariableop*
&savev2_bn_11_gamma_read_readvariableop)
%savev2_bn_11_beta_read_readvariableop0
,savev2_bn_11_moving_mean_read_readvariableop4
0savev2_bn_11_moving_variance_read_readvariableop8
4savev2_conv2dtranspose_12_kernel_read_readvariableop6
2savev2_conv2dtranspose_12_bias_read_readvariableop*
&savev2_bn_12_gamma_read_readvariableop)
%savev2_bn_12_beta_read_readvariableop0
,savev2_bn_12_moving_mean_read_readvariableop4
0savev2_bn_12_moving_variance_read_readvariableop8
4savev2_conv2dtranspose_13_kernel_read_readvariableop6
2savev2_conv2dtranspose_13_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_dense_layer_7_kernel_read_readvariableop-savev2_dense_layer_7_bias_read_readvariableop/savev2_dense_layer_8_kernel_read_readvariableop-savev2_dense_layer_8_bias_read_readvariableop3savev2_conv2dtranspose_9_kernel_read_readvariableop1savev2_conv2dtranspose_9_bias_read_readvariableop%savev2_bn_9_gamma_read_readvariableop$savev2_bn_9_beta_read_readvariableop+savev2_bn_9_moving_mean_read_readvariableop/savev2_bn_9_moving_variance_read_readvariableop4savev2_conv2dtranspose_10_kernel_read_readvariableop2savev2_conv2dtranspose_10_bias_read_readvariableop&savev2_bn_10_gamma_read_readvariableop%savev2_bn_10_beta_read_readvariableop,savev2_bn_10_moving_mean_read_readvariableop0savev2_bn_10_moving_variance_read_readvariableop4savev2_conv2dtranspose_11_kernel_read_readvariableop2savev2_conv2dtranspose_11_bias_read_readvariableop&savev2_bn_11_gamma_read_readvariableop%savev2_bn_11_beta_read_readvariableop,savev2_bn_11_moving_mean_read_readvariableop0savev2_bn_11_moving_variance_read_readvariableop4savev2_conv2dtranspose_12_kernel_read_readvariableop2savev2_conv2dtranspose_12_bias_read_readvariableop&savev2_bn_12_gamma_read_readvariableop%savev2_bn_12_beta_read_readvariableop,savev2_bn_12_moving_mean_read_readvariableop0savev2_bn_12_moving_variance_read_readvariableop4savev2_conv2dtranspose_13_kernel_read_readvariableop2savev2_conv2dtranspose_13_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *-
dtypes#
!2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:
??@:?@:??:?:?:?:?:?:??:?:?:?:?:?:@?:@:@:@:@:@: @: : : : : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??@:!

_output_shapes	
:?@:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!	

_output_shapes	
:?:!


_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:-)
'
_output_shapes
:@?: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: 
?
?
A__inference_BN_12_layer_call_and_return_conditional_losses_404908

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
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
?
?
&__inference_BN_11_layer_call_fn_404754

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
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
GPU2*0J 8? *J
fERC
A__inference_BN_11_layer_call_and_return_conditional_losses_403037?
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
?
?
3__inference_Conv2DTranspose_10_layer_call_fn_404602

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_10_layer_call_and_return_conditional_losses_402899?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_BN_10_layer_call_and_return_conditional_losses_402928

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
C__inference_decoder_layer_call_and_return_conditional_losses_404429

inputs@
,dense_layer_7_matmul_readvariableop_resource:
??<
-dense_layer_7_biasadd_readvariableop_resource:	?@
,dense_layer_8_matmul_readvariableop_resource:
??@<
-dense_layer_8_biasadd_readvariableop_resource:	?@V
:conv2dtranspose_9_conv2d_transpose_readvariableop_resource:??@
1conv2dtranspose_9_biasadd_readvariableop_resource:	?+
bn_9_readvariableop_resource:	?-
bn_9_readvariableop_1_resource:	?<
-bn_9_fusedbatchnormv3_readvariableop_resource:	?>
/bn_9_fusedbatchnormv3_readvariableop_1_resource:	?W
;conv2dtranspose_10_conv2d_transpose_readvariableop_resource:??A
2conv2dtranspose_10_biasadd_readvariableop_resource:	?,
bn_10_readvariableop_resource:	?.
bn_10_readvariableop_1_resource:	?=
.bn_10_fusedbatchnormv3_readvariableop_resource:	??
0bn_10_fusedbatchnormv3_readvariableop_1_resource:	?V
;conv2dtranspose_11_conv2d_transpose_readvariableop_resource:@?@
2conv2dtranspose_11_biasadd_readvariableop_resource:@+
bn_11_readvariableop_resource:@-
bn_11_readvariableop_1_resource:@<
.bn_11_fusedbatchnormv3_readvariableop_resource:@>
0bn_11_fusedbatchnormv3_readvariableop_1_resource:@U
;conv2dtranspose_12_conv2d_transpose_readvariableop_resource: @@
2conv2dtranspose_12_biasadd_readvariableop_resource: +
bn_12_readvariableop_resource: -
bn_12_readvariableop_1_resource: <
.bn_12_fusedbatchnormv3_readvariableop_resource: >
0bn_12_fusedbatchnormv3_readvariableop_1_resource: U
;conv2dtranspose_13_conv2d_transpose_readvariableop_resource: @
2conv2dtranspose_13_biasadd_readvariableop_resource:
identity??BN_10/AssignNewValue?BN_10/AssignNewValue_1?%BN_10/FusedBatchNormV3/ReadVariableOp?'BN_10/FusedBatchNormV3/ReadVariableOp_1?BN_10/ReadVariableOp?BN_10/ReadVariableOp_1?BN_11/AssignNewValue?BN_11/AssignNewValue_1?%BN_11/FusedBatchNormV3/ReadVariableOp?'BN_11/FusedBatchNormV3/ReadVariableOp_1?BN_11/ReadVariableOp?BN_11/ReadVariableOp_1?BN_12/AssignNewValue?BN_12/AssignNewValue_1?%BN_12/FusedBatchNormV3/ReadVariableOp?'BN_12/FusedBatchNormV3/ReadVariableOp_1?BN_12/ReadVariableOp?BN_12/ReadVariableOp_1?BN_9/AssignNewValue?BN_9/AssignNewValue_1?$BN_9/FusedBatchNormV3/ReadVariableOp?&BN_9/FusedBatchNormV3/ReadVariableOp_1?BN_9/ReadVariableOp?BN_9/ReadVariableOp_1?)Conv2DTranspose_10/BiasAdd/ReadVariableOp?2Conv2DTranspose_10/conv2d_transpose/ReadVariableOp?)Conv2DTranspose_11/BiasAdd/ReadVariableOp?2Conv2DTranspose_11/conv2d_transpose/ReadVariableOp?)Conv2DTranspose_12/BiasAdd/ReadVariableOp?2Conv2DTranspose_12/conv2d_transpose/ReadVariableOp?)Conv2DTranspose_13/BiasAdd/ReadVariableOp?2Conv2DTranspose_13/conv2d_transpose/ReadVariableOp?(Conv2DTranspose_9/BiasAdd/ReadVariableOp?1Conv2DTranspose_9/conv2d_transpose/ReadVariableOp?$Dense_Layer_7/BiasAdd/ReadVariableOp?#Dense_Layer_7/MatMul/ReadVariableOp?$Dense_Layer_8/BiasAdd/ReadVariableOp?#Dense_Layer_8/MatMul/ReadVariableOp?
#Dense_Layer_7/MatMul/ReadVariableOpReadVariableOp,dense_layer_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Dense_Layer_7/MatMulMatMulinputs+Dense_Layer_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$Dense_Layer_7/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Dense_Layer_7/BiasAddBiasAddDense_Layer_7/MatMul:product:0,Dense_Layer_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????m
Dense_Layer_7/ReluReluDense_Layer_7/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
#Dense_Layer_8/MatMul/ReadVariableOpReadVariableOp,dense_layer_8_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
Dense_Layer_8/MatMulMatMul Dense_Layer_7/Relu:activations:0+Dense_Layer_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@?
$Dense_Layer_8/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_8_biasadd_readvariableop_resource*
_output_shapes	
:?@*
dtype0?
Dense_Layer_8/BiasAddBiasAddDense_Layer_8/MatMul:product:0,Dense_Layer_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@m
Dense_Layer_8/ReluReluDense_Layer_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????@c
Reshape_Layer/ShapeShape Dense_Layer_8/Relu:activations:0*
T0*
_output_shapes
:k
!Reshape_Layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Reshape_Layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Reshape_Layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Reshape_Layer/strided_sliceStridedSliceReshape_Layer/Shape:output:0*Reshape_Layer/strided_slice/stack:output:0,Reshape_Layer/strided_slice/stack_1:output:0,Reshape_Layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
Reshape_Layer/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :_
Reshape_Layer/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`
Reshape_Layer/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
Reshape_Layer/Reshape/shapePack$Reshape_Layer/strided_slice:output:0&Reshape_Layer/Reshape/shape/1:output:0&Reshape_Layer/Reshape/shape/2:output:0&Reshape_Layer/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
Reshape_Layer/ReshapeReshape Dense_Layer_8/Relu:activations:0$Reshape_Layer/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????e
Conv2DTranspose_9/ShapeShapeReshape_Layer/Reshape:output:0*
T0*
_output_shapes
:o
%Conv2DTranspose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'Conv2DTranspose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'Conv2DTranspose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Conv2DTranspose_9/strided_sliceStridedSlice Conv2DTranspose_9/Shape:output:0.Conv2DTranspose_9/strided_slice/stack:output:00Conv2DTranspose_9/strided_slice/stack_1:output:00Conv2DTranspose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
Conv2DTranspose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :[
Conv2DTranspose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
Conv2DTranspose_9/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
Conv2DTranspose_9/stackPack(Conv2DTranspose_9/strided_slice:output:0"Conv2DTranspose_9/stack/1:output:0"Conv2DTranspose_9/stack/2:output:0"Conv2DTranspose_9/stack/3:output:0*
N*
T0*
_output_shapes
:q
'Conv2DTranspose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)Conv2DTranspose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)Conv2DTranspose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!Conv2DTranspose_9/strided_slice_1StridedSlice Conv2DTranspose_9/stack:output:00Conv2DTranspose_9/strided_slice_1/stack:output:02Conv2DTranspose_9/strided_slice_1/stack_1:output:02Conv2DTranspose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1Conv2DTranspose_9/conv2d_transpose/ReadVariableOpReadVariableOp:conv2dtranspose_9_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
"Conv2DTranspose_9/conv2d_transposeConv2DBackpropInput Conv2DTranspose_9/stack:output:09Conv2DTranspose_9/conv2d_transpose/ReadVariableOp:value:0Reshape_Layer/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
(Conv2DTranspose_9/BiasAdd/ReadVariableOpReadVariableOp1conv2dtranspose_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Conv2DTranspose_9/BiasAddBiasAdd+Conv2DTranspose_9/conv2d_transpose:output:00Conv2DTranspose_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????}
Conv2DTranspose_9/ReluRelu"Conv2DTranspose_9/BiasAdd:output:0*
T0*0
_output_shapes
:??????????m
BN_9/ReadVariableOpReadVariableOpbn_9_readvariableop_resource*
_output_shapes	
:?*
dtype0q
BN_9/ReadVariableOp_1ReadVariableOpbn_9_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
$BN_9/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&BN_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
BN_9/FusedBatchNormV3FusedBatchNormV3$Conv2DTranspose_9/Relu:activations:0BN_9/ReadVariableOp:value:0BN_9/ReadVariableOp_1:value:0,BN_9/FusedBatchNormV3/ReadVariableOp:value:0.BN_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
BN_9/AssignNewValueAssignVariableOp-bn_9_fusedbatchnormv3_readvariableop_resource"BN_9/FusedBatchNormV3:batch_mean:0%^BN_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
BN_9/AssignNewValue_1AssignVariableOp/bn_9_fusedbatchnormv3_readvariableop_1_resource&BN_9/FusedBatchNormV3:batch_variance:0'^BN_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(a
Conv2DTranspose_10/ShapeShapeBN_9/FusedBatchNormV3:y:0*
T0*
_output_shapes
:p
&Conv2DTranspose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(Conv2DTranspose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(Conv2DTranspose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 Conv2DTranspose_10/strided_sliceStridedSlice!Conv2DTranspose_10/Shape:output:0/Conv2DTranspose_10/strided_slice/stack:output:01Conv2DTranspose_10/strided_slice/stack_1:output:01Conv2DTranspose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
Conv2DTranspose_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
Conv2DTranspose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
Conv2DTranspose_10/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
Conv2DTranspose_10/stackPack)Conv2DTranspose_10/strided_slice:output:0#Conv2DTranspose_10/stack/1:output:0#Conv2DTranspose_10/stack/2:output:0#Conv2DTranspose_10/stack/3:output:0*
N*
T0*
_output_shapes
:r
(Conv2DTranspose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Conv2DTranspose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Conv2DTranspose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Conv2DTranspose_10/strided_slice_1StridedSlice!Conv2DTranspose_10/stack:output:01Conv2DTranspose_10/strided_slice_1/stack:output:03Conv2DTranspose_10/strided_slice_1/stack_1:output:03Conv2DTranspose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2Conv2DTranspose_10/conv2d_transpose/ReadVariableOpReadVariableOp;conv2dtranspose_10_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
#Conv2DTranspose_10/conv2d_transposeConv2DBackpropInput!Conv2DTranspose_10/stack:output:0:Conv2DTranspose_10/conv2d_transpose/ReadVariableOp:value:0BN_9/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)Conv2DTranspose_10/BiasAdd/ReadVariableOpReadVariableOp2conv2dtranspose_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Conv2DTranspose_10/BiasAddBiasAdd,Conv2DTranspose_10/conv2d_transpose:output:01Conv2DTranspose_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
Conv2DTranspose_10/ReluRelu#Conv2DTranspose_10/BiasAdd:output:0*
T0*0
_output_shapes
:??????????o
BN_10/ReadVariableOpReadVariableOpbn_10_readvariableop_resource*
_output_shapes	
:?*
dtype0s
BN_10/ReadVariableOp_1ReadVariableOpbn_10_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
%BN_10/FusedBatchNormV3/ReadVariableOpReadVariableOp.bn_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'BN_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0bn_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
BN_10/FusedBatchNormV3FusedBatchNormV3%Conv2DTranspose_10/Relu:activations:0BN_10/ReadVariableOp:value:0BN_10/ReadVariableOp_1:value:0-BN_10/FusedBatchNormV3/ReadVariableOp:value:0/BN_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
BN_10/AssignNewValueAssignVariableOp.bn_10_fusedbatchnormv3_readvariableop_resource#BN_10/FusedBatchNormV3:batch_mean:0&^BN_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
BN_10/AssignNewValue_1AssignVariableOp0bn_10_fusedbatchnormv3_readvariableop_1_resource'BN_10/FusedBatchNormV3:batch_variance:0(^BN_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(b
Conv2DTranspose_11/ShapeShapeBN_10/FusedBatchNormV3:y:0*
T0*
_output_shapes
:p
&Conv2DTranspose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(Conv2DTranspose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(Conv2DTranspose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 Conv2DTranspose_11/strided_sliceStridedSlice!Conv2DTranspose_11/Shape:output:0/Conv2DTranspose_11/strided_slice/stack:output:01Conv2DTranspose_11/strided_slice/stack_1:output:01Conv2DTranspose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
Conv2DTranspose_11/stack/1Const*
_output_shapes
: *
dtype0*
value	B : \
Conv2DTranspose_11/stack/2Const*
_output_shapes
: *
dtype0*
value	B : \
Conv2DTranspose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
Conv2DTranspose_11/stackPack)Conv2DTranspose_11/strided_slice:output:0#Conv2DTranspose_11/stack/1:output:0#Conv2DTranspose_11/stack/2:output:0#Conv2DTranspose_11/stack/3:output:0*
N*
T0*
_output_shapes
:r
(Conv2DTranspose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Conv2DTranspose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Conv2DTranspose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Conv2DTranspose_11/strided_slice_1StridedSlice!Conv2DTranspose_11/stack:output:01Conv2DTranspose_11/strided_slice_1/stack:output:03Conv2DTranspose_11/strided_slice_1/stack_1:output:03Conv2DTranspose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2Conv2DTranspose_11/conv2d_transpose/ReadVariableOpReadVariableOp;conv2dtranspose_11_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
#Conv2DTranspose_11/conv2d_transposeConv2DBackpropInput!Conv2DTranspose_11/stack:output:0:Conv2DTranspose_11/conv2d_transpose/ReadVariableOp:value:0BN_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
)Conv2DTranspose_11/BiasAdd/ReadVariableOpReadVariableOp2conv2dtranspose_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Conv2DTranspose_11/BiasAddBiasAdd,Conv2DTranspose_11/conv2d_transpose:output:01Conv2DTranspose_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @~
Conv2DTranspose_11/ReluRelu#Conv2DTranspose_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @n
BN_11/ReadVariableOpReadVariableOpbn_11_readvariableop_resource*
_output_shapes
:@*
dtype0r
BN_11/ReadVariableOp_1ReadVariableOpbn_11_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
%BN_11/FusedBatchNormV3/ReadVariableOpReadVariableOp.bn_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
'BN_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0bn_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
BN_11/FusedBatchNormV3FusedBatchNormV3%Conv2DTranspose_11/Relu:activations:0BN_11/ReadVariableOp:value:0BN_11/ReadVariableOp_1:value:0-BN_11/FusedBatchNormV3/ReadVariableOp:value:0/BN_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
BN_11/AssignNewValueAssignVariableOp.bn_11_fusedbatchnormv3_readvariableop_resource#BN_11/FusedBatchNormV3:batch_mean:0&^BN_11/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
BN_11/AssignNewValue_1AssignVariableOp0bn_11_fusedbatchnormv3_readvariableop_1_resource'BN_11/FusedBatchNormV3:batch_variance:0(^BN_11/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(b
Conv2DTranspose_12/ShapeShapeBN_11/FusedBatchNormV3:y:0*
T0*
_output_shapes
:p
&Conv2DTranspose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(Conv2DTranspose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(Conv2DTranspose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 Conv2DTranspose_12/strided_sliceStridedSlice!Conv2DTranspose_12/Shape:output:0/Conv2DTranspose_12/strided_slice/stack:output:01Conv2DTranspose_12/strided_slice/stack_1:output:01Conv2DTranspose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
Conv2DTranspose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
Conv2DTranspose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
Conv2DTranspose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
Conv2DTranspose_12/stackPack)Conv2DTranspose_12/strided_slice:output:0#Conv2DTranspose_12/stack/1:output:0#Conv2DTranspose_12/stack/2:output:0#Conv2DTranspose_12/stack/3:output:0*
N*
T0*
_output_shapes
:r
(Conv2DTranspose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Conv2DTranspose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Conv2DTranspose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Conv2DTranspose_12/strided_slice_1StridedSlice!Conv2DTranspose_12/stack:output:01Conv2DTranspose_12/strided_slice_1/stack:output:03Conv2DTranspose_12/strided_slice_1/stack_1:output:03Conv2DTranspose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2Conv2DTranspose_12/conv2d_transpose/ReadVariableOpReadVariableOp;conv2dtranspose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
#Conv2DTranspose_12/conv2d_transposeConv2DBackpropInput!Conv2DTranspose_12/stack:output:0:Conv2DTranspose_12/conv2d_transpose/ReadVariableOp:value:0BN_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
)Conv2DTranspose_12/BiasAdd/ReadVariableOpReadVariableOp2conv2dtranspose_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Conv2DTranspose_12/BiasAddBiasAdd,Conv2DTranspose_12/conv2d_transpose:output:01Conv2DTranspose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ~
Conv2DTranspose_12/ReluRelu#Conv2DTranspose_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ n
BN_12/ReadVariableOpReadVariableOpbn_12_readvariableop_resource*
_output_shapes
: *
dtype0r
BN_12/ReadVariableOp_1ReadVariableOpbn_12_readvariableop_1_resource*
_output_shapes
: *
dtype0?
%BN_12/FusedBatchNormV3/ReadVariableOpReadVariableOp.bn_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
'BN_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0bn_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
BN_12/FusedBatchNormV3FusedBatchNormV3%Conv2DTranspose_12/Relu:activations:0BN_12/ReadVariableOp:value:0BN_12/ReadVariableOp_1:value:0-BN_12/FusedBatchNormV3/ReadVariableOp:value:0/BN_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
BN_12/AssignNewValueAssignVariableOp.bn_12_fusedbatchnormv3_readvariableop_resource#BN_12/FusedBatchNormV3:batch_mean:0&^BN_12/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
BN_12/AssignNewValue_1AssignVariableOp0bn_12_fusedbatchnormv3_readvariableop_1_resource'BN_12/FusedBatchNormV3:batch_variance:0(^BN_12/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(b
Conv2DTranspose_13/ShapeShapeBN_12/FusedBatchNormV3:y:0*
T0*
_output_shapes
:p
&Conv2DTranspose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(Conv2DTranspose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(Conv2DTranspose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 Conv2DTranspose_13/strided_sliceStridedSlice!Conv2DTranspose_13/Shape:output:0/Conv2DTranspose_13/strided_slice/stack:output:01Conv2DTranspose_13/strided_slice/stack_1:output:01Conv2DTranspose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
Conv2DTranspose_13/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?]
Conv2DTranspose_13/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?\
Conv2DTranspose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
Conv2DTranspose_13/stackPack)Conv2DTranspose_13/strided_slice:output:0#Conv2DTranspose_13/stack/1:output:0#Conv2DTranspose_13/stack/2:output:0#Conv2DTranspose_13/stack/3:output:0*
N*
T0*
_output_shapes
:r
(Conv2DTranspose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Conv2DTranspose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Conv2DTranspose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Conv2DTranspose_13/strided_slice_1StridedSlice!Conv2DTranspose_13/stack:output:01Conv2DTranspose_13/strided_slice_1/stack:output:03Conv2DTranspose_13/strided_slice_1/stack_1:output:03Conv2DTranspose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2Conv2DTranspose_13/conv2d_transpose/ReadVariableOpReadVariableOp;conv2dtranspose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
#Conv2DTranspose_13/conv2d_transposeConv2DBackpropInput!Conv2DTranspose_13/stack:output:0:Conv2DTranspose_13/conv2d_transpose/ReadVariableOp:value:0BN_12/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
)Conv2DTranspose_13/BiasAdd/ReadVariableOpReadVariableOp2conv2dtranspose_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Conv2DTranspose_13/BiasAddBiasAdd,Conv2DTranspose_13/conv2d_transpose:output:01Conv2DTranspose_13/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
Conv2DTranspose_13/TanhTanh#Conv2DTranspose_13/BiasAdd:output:0*
T0*1
_output_shapes
:???????????t
IdentityIdentityConv2DTranspose_13/Tanh:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp^BN_10/AssignNewValue^BN_10/AssignNewValue_1&^BN_10/FusedBatchNormV3/ReadVariableOp(^BN_10/FusedBatchNormV3/ReadVariableOp_1^BN_10/ReadVariableOp^BN_10/ReadVariableOp_1^BN_11/AssignNewValue^BN_11/AssignNewValue_1&^BN_11/FusedBatchNormV3/ReadVariableOp(^BN_11/FusedBatchNormV3/ReadVariableOp_1^BN_11/ReadVariableOp^BN_11/ReadVariableOp_1^BN_12/AssignNewValue^BN_12/AssignNewValue_1&^BN_12/FusedBatchNormV3/ReadVariableOp(^BN_12/FusedBatchNormV3/ReadVariableOp_1^BN_12/ReadVariableOp^BN_12/ReadVariableOp_1^BN_9/AssignNewValue^BN_9/AssignNewValue_1%^BN_9/FusedBatchNormV3/ReadVariableOp'^BN_9/FusedBatchNormV3/ReadVariableOp_1^BN_9/ReadVariableOp^BN_9/ReadVariableOp_1*^Conv2DTranspose_10/BiasAdd/ReadVariableOp3^Conv2DTranspose_10/conv2d_transpose/ReadVariableOp*^Conv2DTranspose_11/BiasAdd/ReadVariableOp3^Conv2DTranspose_11/conv2d_transpose/ReadVariableOp*^Conv2DTranspose_12/BiasAdd/ReadVariableOp3^Conv2DTranspose_12/conv2d_transpose/ReadVariableOp*^Conv2DTranspose_13/BiasAdd/ReadVariableOp3^Conv2DTranspose_13/conv2d_transpose/ReadVariableOp)^Conv2DTranspose_9/BiasAdd/ReadVariableOp2^Conv2DTranspose_9/conv2d_transpose/ReadVariableOp%^Dense_Layer_7/BiasAdd/ReadVariableOp$^Dense_Layer_7/MatMul/ReadVariableOp%^Dense_Layer_8/BiasAdd/ReadVariableOp$^Dense_Layer_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2,
BN_10/AssignNewValueBN_10/AssignNewValue20
BN_10/AssignNewValue_1BN_10/AssignNewValue_12N
%BN_10/FusedBatchNormV3/ReadVariableOp%BN_10/FusedBatchNormV3/ReadVariableOp2R
'BN_10/FusedBatchNormV3/ReadVariableOp_1'BN_10/FusedBatchNormV3/ReadVariableOp_12,
BN_10/ReadVariableOpBN_10/ReadVariableOp20
BN_10/ReadVariableOp_1BN_10/ReadVariableOp_12,
BN_11/AssignNewValueBN_11/AssignNewValue20
BN_11/AssignNewValue_1BN_11/AssignNewValue_12N
%BN_11/FusedBatchNormV3/ReadVariableOp%BN_11/FusedBatchNormV3/ReadVariableOp2R
'BN_11/FusedBatchNormV3/ReadVariableOp_1'BN_11/FusedBatchNormV3/ReadVariableOp_12,
BN_11/ReadVariableOpBN_11/ReadVariableOp20
BN_11/ReadVariableOp_1BN_11/ReadVariableOp_12,
BN_12/AssignNewValueBN_12/AssignNewValue20
BN_12/AssignNewValue_1BN_12/AssignNewValue_12N
%BN_12/FusedBatchNormV3/ReadVariableOp%BN_12/FusedBatchNormV3/ReadVariableOp2R
'BN_12/FusedBatchNormV3/ReadVariableOp_1'BN_12/FusedBatchNormV3/ReadVariableOp_12,
BN_12/ReadVariableOpBN_12/ReadVariableOp20
BN_12/ReadVariableOp_1BN_12/ReadVariableOp_12*
BN_9/AssignNewValueBN_9/AssignNewValue2.
BN_9/AssignNewValue_1BN_9/AssignNewValue_12L
$BN_9/FusedBatchNormV3/ReadVariableOp$BN_9/FusedBatchNormV3/ReadVariableOp2P
&BN_9/FusedBatchNormV3/ReadVariableOp_1&BN_9/FusedBatchNormV3/ReadVariableOp_12*
BN_9/ReadVariableOpBN_9/ReadVariableOp2.
BN_9/ReadVariableOp_1BN_9/ReadVariableOp_12V
)Conv2DTranspose_10/BiasAdd/ReadVariableOp)Conv2DTranspose_10/BiasAdd/ReadVariableOp2h
2Conv2DTranspose_10/conv2d_transpose/ReadVariableOp2Conv2DTranspose_10/conv2d_transpose/ReadVariableOp2V
)Conv2DTranspose_11/BiasAdd/ReadVariableOp)Conv2DTranspose_11/BiasAdd/ReadVariableOp2h
2Conv2DTranspose_11/conv2d_transpose/ReadVariableOp2Conv2DTranspose_11/conv2d_transpose/ReadVariableOp2V
)Conv2DTranspose_12/BiasAdd/ReadVariableOp)Conv2DTranspose_12/BiasAdd/ReadVariableOp2h
2Conv2DTranspose_12/conv2d_transpose/ReadVariableOp2Conv2DTranspose_12/conv2d_transpose/ReadVariableOp2V
)Conv2DTranspose_13/BiasAdd/ReadVariableOp)Conv2DTranspose_13/BiasAdd/ReadVariableOp2h
2Conv2DTranspose_13/conv2d_transpose/ReadVariableOp2Conv2DTranspose_13/conv2d_transpose/ReadVariableOp2T
(Conv2DTranspose_9/BiasAdd/ReadVariableOp(Conv2DTranspose_9/BiasAdd/ReadVariableOp2f
1Conv2DTranspose_9/conv2d_transpose/ReadVariableOp1Conv2DTranspose_9/conv2d_transpose/ReadVariableOp2L
$Dense_Layer_7/BiasAdd/ReadVariableOp$Dense_Layer_7/BiasAdd/ReadVariableOp2J
#Dense_Layer_7/MatMul/ReadVariableOp#Dense_Layer_7/MatMul/ReadVariableOp2L
$Dense_Layer_8/BiasAdd/ReadVariableOp$Dense_Layer_8/BiasAdd/ReadVariableOp2J
#Dense_Layer_8/MatMul/ReadVariableOp#Dense_Layer_8/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
I__inference_Dense_Layer_8_layer_call_and_return_conditional_losses_404469

inputs2
matmul_readvariableop_resource:
??@.
biasadd_readvariableop_resource:	?@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?@*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????@b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
I__inference_Dense_Layer_7_layer_call_and_return_conditional_losses_403251

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?C
?
C__inference_decoder_layer_call_and_return_conditional_losses_403788
decoder_input(
dense_layer_7_403715:
??#
dense_layer_7_403717:	?(
dense_layer_8_403720:
??@#
dense_layer_8_403722:	?@4
conv2dtranspose_9_403726:??'
conv2dtranspose_9_403728:	?
bn_9_403731:	?
bn_9_403733:	?
bn_9_403735:	?
bn_9_403737:	?5
conv2dtranspose_10_403740:??(
conv2dtranspose_10_403742:	?
bn_10_403745:	?
bn_10_403747:	?
bn_10_403749:	?
bn_10_403751:	?4
conv2dtranspose_11_403754:@?'
conv2dtranspose_11_403756:@
bn_11_403759:@
bn_11_403761:@
bn_11_403763:@
bn_11_403765:@3
conv2dtranspose_12_403768: @'
conv2dtranspose_12_403770: 
bn_12_403773: 
bn_12_403775: 
bn_12_403777: 
bn_12_403779: 3
conv2dtranspose_13_403782: '
conv2dtranspose_13_403784:
identity??BN_10/StatefulPartitionedCall?BN_11/StatefulPartitionedCall?BN_12/StatefulPartitionedCall?BN_9/StatefulPartitionedCall?*Conv2DTranspose_10/StatefulPartitionedCall?*Conv2DTranspose_11/StatefulPartitionedCall?*Conv2DTranspose_12/StatefulPartitionedCall?*Conv2DTranspose_13/StatefulPartitionedCall?)Conv2DTranspose_9/StatefulPartitionedCall?%Dense_Layer_7/StatefulPartitionedCall?%Dense_Layer_8/StatefulPartitionedCall?
%Dense_Layer_7/StatefulPartitionedCallStatefulPartitionedCalldecoder_inputdense_layer_7_403715dense_layer_7_403717*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Dense_Layer_7_layer_call_and_return_conditional_losses_403251?
%Dense_Layer_8/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_7/StatefulPartitionedCall:output:0dense_layer_8_403720dense_layer_8_403722*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Dense_Layer_8_layer_call_and_return_conditional_losses_403268?
Reshape_Layer/PartitionedCallPartitionedCall.Dense_Layer_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Reshape_Layer_layer_call_and_return_conditional_losses_403288?
)Conv2DTranspose_9/StatefulPartitionedCallStatefulPartitionedCall&Reshape_Layer/PartitionedCall:output:0conv2dtranspose_9_403726conv2dtranspose_9_403728*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_Conv2DTranspose_9_layer_call_and_return_conditional_losses_402790?
BN_9/StatefulPartitionedCallStatefulPartitionedCall2Conv2DTranspose_9/StatefulPartitionedCall:output:0bn_9_403731bn_9_403733bn_9_403735bn_9_403737*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_BN_9_layer_call_and_return_conditional_losses_402819?
*Conv2DTranspose_10/StatefulPartitionedCallStatefulPartitionedCall%BN_9/StatefulPartitionedCall:output:0conv2dtranspose_10_403740conv2dtranspose_10_403742*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_10_layer_call_and_return_conditional_losses_402899?
BN_10/StatefulPartitionedCallStatefulPartitionedCall3Conv2DTranspose_10/StatefulPartitionedCall:output:0bn_10_403745bn_10_403747bn_10_403749bn_10_403751*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_BN_10_layer_call_and_return_conditional_losses_402928?
*Conv2DTranspose_11/StatefulPartitionedCallStatefulPartitionedCall&BN_10/StatefulPartitionedCall:output:0conv2dtranspose_11_403754conv2dtranspose_11_403756*
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
GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_11_layer_call_and_return_conditional_losses_403008?
BN_11/StatefulPartitionedCallStatefulPartitionedCall3Conv2DTranspose_11/StatefulPartitionedCall:output:0bn_11_403759bn_11_403761bn_11_403763bn_11_403765*
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
GPU2*0J 8? *J
fERC
A__inference_BN_11_layer_call_and_return_conditional_losses_403037?
*Conv2DTranspose_12/StatefulPartitionedCallStatefulPartitionedCall&BN_11/StatefulPartitionedCall:output:0conv2dtranspose_12_403768conv2dtranspose_12_403770*
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
GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_12_layer_call_and_return_conditional_losses_403117?
BN_12/StatefulPartitionedCallStatefulPartitionedCall3Conv2DTranspose_12/StatefulPartitionedCall:output:0bn_12_403773bn_12_403775bn_12_403777bn_12_403779*
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
GPU2*0J 8? *J
fERC
A__inference_BN_12_layer_call_and_return_conditional_losses_403146?
*Conv2DTranspose_13/StatefulPartitionedCallStatefulPartitionedCall&BN_12/StatefulPartitionedCall:output:0conv2dtranspose_13_403782conv2dtranspose_13_403784*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_13_layer_call_and_return_conditional_losses_403226?
IdentityIdentity3Conv2DTranspose_13/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp^BN_10/StatefulPartitionedCall^BN_11/StatefulPartitionedCall^BN_12/StatefulPartitionedCall^BN_9/StatefulPartitionedCall+^Conv2DTranspose_10/StatefulPartitionedCall+^Conv2DTranspose_11/StatefulPartitionedCall+^Conv2DTranspose_12/StatefulPartitionedCall+^Conv2DTranspose_13/StatefulPartitionedCall*^Conv2DTranspose_9/StatefulPartitionedCall&^Dense_Layer_7/StatefulPartitionedCall&^Dense_Layer_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
BN_10/StatefulPartitionedCallBN_10/StatefulPartitionedCall2>
BN_11/StatefulPartitionedCallBN_11/StatefulPartitionedCall2>
BN_12/StatefulPartitionedCallBN_12/StatefulPartitionedCall2<
BN_9/StatefulPartitionedCallBN_9/StatefulPartitionedCall2X
*Conv2DTranspose_10/StatefulPartitionedCall*Conv2DTranspose_10/StatefulPartitionedCall2X
*Conv2DTranspose_11/StatefulPartitionedCall*Conv2DTranspose_11/StatefulPartitionedCall2X
*Conv2DTranspose_12/StatefulPartitionedCall*Conv2DTranspose_12/StatefulPartitionedCall2X
*Conv2DTranspose_13/StatefulPartitionedCall*Conv2DTranspose_13/StatefulPartitionedCall2V
)Conv2DTranspose_9/StatefulPartitionedCall)Conv2DTranspose_9/StatefulPartitionedCall2N
%Dense_Layer_7/StatefulPartitionedCall%Dense_Layer_7/StatefulPartitionedCall2N
%Dense_Layer_8/StatefulPartitionedCall%Dense_Layer_8/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameDecoder_Input
?
?
3__inference_Conv2DTranspose_11_layer_call_fn_404707

inputs"
unknown:@?
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
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
GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_11_layer_call_and_return_conditional_losses_403008?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?!
?
N__inference_Conv2DTranspose_10_layer_call_and_return_conditional_losses_404636

inputsD
(conv2d_transpose_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????k
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_BN_11_layer_call_and_return_conditional_losses_403037

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
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
?!
?
N__inference_Conv2DTranspose_13_layer_call_and_return_conditional_losses_404951

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
A__inference_BN_10_layer_call_and_return_conditional_losses_404680

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_BN_10_layer_call_fn_404662

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_BN_10_layer_call_and_return_conditional_losses_402959?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_decoder_layer_call_fn_403712
decoder_input
unknown:
??
	unknown_0:	?
	unknown_1:
??@
	unknown_2:	?@%
	unknown_3:??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?%

unknown_15:@?

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@$

unknown_21: @

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: $

unknown_27: 

unknown_28:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldecoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*8
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_403584y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameDecoder_Input
?
?
A__inference_BN_11_layer_call_and_return_conditional_losses_403068

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
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
?!
?
N__inference_Conv2DTranspose_11_layer_call_and_return_conditional_losses_404741

inputsC
(conv2d_transpose_readvariableop_resource:@?-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_BN_11_layer_call_fn_404767

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
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
GPU2*0J 8? *J
fERC
A__inference_BN_11_layer_call_and_return_conditional_losses_403068?
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
?
?
.__inference_Dense_Layer_7_layer_call_fn_404438

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Dense_Layer_7_layer_call_and_return_conditional_losses_403251p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
N__inference_Conv2DTranspose_11_layer_call_and_return_conditional_losses_403008

inputsC
(conv2d_transpose_readvariableop_resource:@?-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
I__inference_Dense_Layer_8_layer_call_and_return_conditional_losses_403268

inputs2
matmul_readvariableop_resource:
??@.
biasadd_readvariableop_resource:	?@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?@*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????@b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
N__inference_Conv2DTranspose_12_layer_call_and_return_conditional_losses_403117

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
(__inference_decoder_layer_call_fn_403996

inputs
unknown:
??
	unknown_0:	?
	unknown_1:
??@
	unknown_2:	?@%
	unknown_3:??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?%

unknown_15:@?

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@$

unknown_21: @

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: $

unknown_27: 

unknown_28:
identity??StatefulPartitionedCall?
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_403352y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?w
?
"__inference__traced_restore_405164
file_prefix9
%assignvariableop_dense_layer_7_kernel:
??4
%assignvariableop_1_dense_layer_7_bias:	?;
'assignvariableop_2_dense_layer_8_kernel:
??@4
%assignvariableop_3_dense_layer_8_bias:	?@G
+assignvariableop_4_conv2dtranspose_9_kernel:??8
)assignvariableop_5_conv2dtranspose_9_bias:	?,
assignvariableop_6_bn_9_gamma:	?+
assignvariableop_7_bn_9_beta:	?2
#assignvariableop_8_bn_9_moving_mean:	?6
'assignvariableop_9_bn_9_moving_variance:	?I
-assignvariableop_10_conv2dtranspose_10_kernel:??:
+assignvariableop_11_conv2dtranspose_10_bias:	?.
assignvariableop_12_bn_10_gamma:	?-
assignvariableop_13_bn_10_beta:	?4
%assignvariableop_14_bn_10_moving_mean:	?8
)assignvariableop_15_bn_10_moving_variance:	?H
-assignvariableop_16_conv2dtranspose_11_kernel:@?9
+assignvariableop_17_conv2dtranspose_11_bias:@-
assignvariableop_18_bn_11_gamma:@,
assignvariableop_19_bn_11_beta:@3
%assignvariableop_20_bn_11_moving_mean:@7
)assignvariableop_21_bn_11_moving_variance:@G
-assignvariableop_22_conv2dtranspose_12_kernel: @9
+assignvariableop_23_conv2dtranspose_12_bias: -
assignvariableop_24_bn_12_gamma: ,
assignvariableop_25_bn_12_beta: 3
%assignvariableop_26_bn_12_moving_mean: 7
)assignvariableop_27_bn_12_moving_variance: G
-assignvariableop_28_conv2dtranspose_13_kernel: 9
+assignvariableop_29_conv2dtranspose_13_bias:
identity_31??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp%assignvariableop_dense_layer_7_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp%assignvariableop_1_dense_layer_7_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp'assignvariableop_2_dense_layer_8_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp%assignvariableop_3_dense_layer_8_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp+assignvariableop_4_conv2dtranspose_9_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp)assignvariableop_5_conv2dtranspose_9_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_bn_9_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_bn_9_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_bn_9_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp'assignvariableop_9_bn_9_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp-assignvariableop_10_conv2dtranspose_10_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_conv2dtranspose_10_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_bn_10_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_bn_10_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp%assignvariableop_14_bn_10_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp)assignvariableop_15_bn_10_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp-assignvariableop_16_conv2dtranspose_11_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp+assignvariableop_17_conv2dtranspose_11_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_bn_11_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_bn_11_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp%assignvariableop_20_bn_11_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_bn_11_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp-assignvariableop_22_conv2dtranspose_12_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_conv2dtranspose_12_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpassignvariableop_24_bn_12_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpassignvariableop_25_bn_12_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp%assignvariableop_26_bn_12_moving_meanIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp)assignvariableop_27_bn_12_moving_varianceIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp-assignvariableop_28_conv2dtranspose_13_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_conv2dtranspose_13_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_3AssignVariableOp_32(
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
?
?
A__inference_BN_10_layer_call_and_return_conditional_losses_404698

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?!
?
M__inference_Conv2DTranspose_9_layer_call_and_return_conditional_losses_402790

inputsD
(conv2d_transpose_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????k
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?C
?
C__inference_decoder_layer_call_and_return_conditional_losses_403352

inputs(
dense_layer_7_403252:
??#
dense_layer_7_403254:	?(
dense_layer_8_403269:
??@#
dense_layer_8_403271:	?@4
conv2dtranspose_9_403290:??'
conv2dtranspose_9_403292:	?
bn_9_403295:	?
bn_9_403297:	?
bn_9_403299:	?
bn_9_403301:	?5
conv2dtranspose_10_403304:??(
conv2dtranspose_10_403306:	?
bn_10_403309:	?
bn_10_403311:	?
bn_10_403313:	?
bn_10_403315:	?4
conv2dtranspose_11_403318:@?'
conv2dtranspose_11_403320:@
bn_11_403323:@
bn_11_403325:@
bn_11_403327:@
bn_11_403329:@3
conv2dtranspose_12_403332: @'
conv2dtranspose_12_403334: 
bn_12_403337: 
bn_12_403339: 
bn_12_403341: 
bn_12_403343: 3
conv2dtranspose_13_403346: '
conv2dtranspose_13_403348:
identity??BN_10/StatefulPartitionedCall?BN_11/StatefulPartitionedCall?BN_12/StatefulPartitionedCall?BN_9/StatefulPartitionedCall?*Conv2DTranspose_10/StatefulPartitionedCall?*Conv2DTranspose_11/StatefulPartitionedCall?*Conv2DTranspose_12/StatefulPartitionedCall?*Conv2DTranspose_13/StatefulPartitionedCall?)Conv2DTranspose_9/StatefulPartitionedCall?%Dense_Layer_7/StatefulPartitionedCall?%Dense_Layer_8/StatefulPartitionedCall?
%Dense_Layer_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_layer_7_403252dense_layer_7_403254*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Dense_Layer_7_layer_call_and_return_conditional_losses_403251?
%Dense_Layer_8/StatefulPartitionedCallStatefulPartitionedCall.Dense_Layer_7/StatefulPartitionedCall:output:0dense_layer_8_403269dense_layer_8_403271*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Dense_Layer_8_layer_call_and_return_conditional_losses_403268?
Reshape_Layer/PartitionedCallPartitionedCall.Dense_Layer_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Reshape_Layer_layer_call_and_return_conditional_losses_403288?
)Conv2DTranspose_9/StatefulPartitionedCallStatefulPartitionedCall&Reshape_Layer/PartitionedCall:output:0conv2dtranspose_9_403290conv2dtranspose_9_403292*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_Conv2DTranspose_9_layer_call_and_return_conditional_losses_402790?
BN_9/StatefulPartitionedCallStatefulPartitionedCall2Conv2DTranspose_9/StatefulPartitionedCall:output:0bn_9_403295bn_9_403297bn_9_403299bn_9_403301*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_BN_9_layer_call_and_return_conditional_losses_402819?
*Conv2DTranspose_10/StatefulPartitionedCallStatefulPartitionedCall%BN_9/StatefulPartitionedCall:output:0conv2dtranspose_10_403304conv2dtranspose_10_403306*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_10_layer_call_and_return_conditional_losses_402899?
BN_10/StatefulPartitionedCallStatefulPartitionedCall3Conv2DTranspose_10/StatefulPartitionedCall:output:0bn_10_403309bn_10_403311bn_10_403313bn_10_403315*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_BN_10_layer_call_and_return_conditional_losses_402928?
*Conv2DTranspose_11/StatefulPartitionedCallStatefulPartitionedCall&BN_10/StatefulPartitionedCall:output:0conv2dtranspose_11_403318conv2dtranspose_11_403320*
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
GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_11_layer_call_and_return_conditional_losses_403008?
BN_11/StatefulPartitionedCallStatefulPartitionedCall3Conv2DTranspose_11/StatefulPartitionedCall:output:0bn_11_403323bn_11_403325bn_11_403327bn_11_403329*
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
GPU2*0J 8? *J
fERC
A__inference_BN_11_layer_call_and_return_conditional_losses_403037?
*Conv2DTranspose_12/StatefulPartitionedCallStatefulPartitionedCall&BN_11/StatefulPartitionedCall:output:0conv2dtranspose_12_403332conv2dtranspose_12_403334*
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
GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_12_layer_call_and_return_conditional_losses_403117?
BN_12/StatefulPartitionedCallStatefulPartitionedCall3Conv2DTranspose_12/StatefulPartitionedCall:output:0bn_12_403337bn_12_403339bn_12_403341bn_12_403343*
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
GPU2*0J 8? *J
fERC
A__inference_BN_12_layer_call_and_return_conditional_losses_403146?
*Conv2DTranspose_13/StatefulPartitionedCallStatefulPartitionedCall&BN_12/StatefulPartitionedCall:output:0conv2dtranspose_13_403346conv2dtranspose_13_403348*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_13_layer_call_and_return_conditional_losses_403226?
IdentityIdentity3Conv2DTranspose_13/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp^BN_10/StatefulPartitionedCall^BN_11/StatefulPartitionedCall^BN_12/StatefulPartitionedCall^BN_9/StatefulPartitionedCall+^Conv2DTranspose_10/StatefulPartitionedCall+^Conv2DTranspose_11/StatefulPartitionedCall+^Conv2DTranspose_12/StatefulPartitionedCall+^Conv2DTranspose_13/StatefulPartitionedCall*^Conv2DTranspose_9/StatefulPartitionedCall&^Dense_Layer_7/StatefulPartitionedCall&^Dense_Layer_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
BN_10/StatefulPartitionedCallBN_10/StatefulPartitionedCall2>
BN_11/StatefulPartitionedCallBN_11/StatefulPartitionedCall2>
BN_12/StatefulPartitionedCallBN_12/StatefulPartitionedCall2<
BN_9/StatefulPartitionedCallBN_9/StatefulPartitionedCall2X
*Conv2DTranspose_10/StatefulPartitionedCall*Conv2DTranspose_10/StatefulPartitionedCall2X
*Conv2DTranspose_11/StatefulPartitionedCall*Conv2DTranspose_11/StatefulPartitionedCall2X
*Conv2DTranspose_12/StatefulPartitionedCall*Conv2DTranspose_12/StatefulPartitionedCall2X
*Conv2DTranspose_13/StatefulPartitionedCall*Conv2DTranspose_13/StatefulPartitionedCall2V
)Conv2DTranspose_9/StatefulPartitionedCall)Conv2DTranspose_9/StatefulPartitionedCall2N
%Dense_Layer_7/StatefulPartitionedCall%Dense_Layer_7/StatefulPartitionedCall2N
%Dense_Layer_8/StatefulPartitionedCall%Dense_Layer_8/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_BN_10_layer_call_fn_404649

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_BN_10_layer_call_and_return_conditional_losses_402928?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
@__inference_BN_9_layer_call_and_return_conditional_losses_402819

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_Reshape_Layer_layer_call_fn_404474

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_Reshape_Layer_layer_call_and_return_conditional_losses_403288i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????@:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
e
I__inference_Reshape_Layer_layer_call_and_return_conditional_losses_404488

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :R
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????a
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????@:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_402752
decoder_inputH
4decoder_dense_layer_7_matmul_readvariableop_resource:
??D
5decoder_dense_layer_7_biasadd_readvariableop_resource:	?H
4decoder_dense_layer_8_matmul_readvariableop_resource:
??@D
5decoder_dense_layer_8_biasadd_readvariableop_resource:	?@^
Bdecoder_conv2dtranspose_9_conv2d_transpose_readvariableop_resource:??H
9decoder_conv2dtranspose_9_biasadd_readvariableop_resource:	?3
$decoder_bn_9_readvariableop_resource:	?5
&decoder_bn_9_readvariableop_1_resource:	?D
5decoder_bn_9_fusedbatchnormv3_readvariableop_resource:	?F
7decoder_bn_9_fusedbatchnormv3_readvariableop_1_resource:	?_
Cdecoder_conv2dtranspose_10_conv2d_transpose_readvariableop_resource:??I
:decoder_conv2dtranspose_10_biasadd_readvariableop_resource:	?4
%decoder_bn_10_readvariableop_resource:	?6
'decoder_bn_10_readvariableop_1_resource:	?E
6decoder_bn_10_fusedbatchnormv3_readvariableop_resource:	?G
8decoder_bn_10_fusedbatchnormv3_readvariableop_1_resource:	?^
Cdecoder_conv2dtranspose_11_conv2d_transpose_readvariableop_resource:@?H
:decoder_conv2dtranspose_11_biasadd_readvariableop_resource:@3
%decoder_bn_11_readvariableop_resource:@5
'decoder_bn_11_readvariableop_1_resource:@D
6decoder_bn_11_fusedbatchnormv3_readvariableop_resource:@F
8decoder_bn_11_fusedbatchnormv3_readvariableop_1_resource:@]
Cdecoder_conv2dtranspose_12_conv2d_transpose_readvariableop_resource: @H
:decoder_conv2dtranspose_12_biasadd_readvariableop_resource: 3
%decoder_bn_12_readvariableop_resource: 5
'decoder_bn_12_readvariableop_1_resource: D
6decoder_bn_12_fusedbatchnormv3_readvariableop_resource: F
8decoder_bn_12_fusedbatchnormv3_readvariableop_1_resource: ]
Cdecoder_conv2dtranspose_13_conv2d_transpose_readvariableop_resource: H
:decoder_conv2dtranspose_13_biasadd_readvariableop_resource:
identity??-decoder/BN_10/FusedBatchNormV3/ReadVariableOp?/decoder/BN_10/FusedBatchNormV3/ReadVariableOp_1?decoder/BN_10/ReadVariableOp?decoder/BN_10/ReadVariableOp_1?-decoder/BN_11/FusedBatchNormV3/ReadVariableOp?/decoder/BN_11/FusedBatchNormV3/ReadVariableOp_1?decoder/BN_11/ReadVariableOp?decoder/BN_11/ReadVariableOp_1?-decoder/BN_12/FusedBatchNormV3/ReadVariableOp?/decoder/BN_12/FusedBatchNormV3/ReadVariableOp_1?decoder/BN_12/ReadVariableOp?decoder/BN_12/ReadVariableOp_1?,decoder/BN_9/FusedBatchNormV3/ReadVariableOp?.decoder/BN_9/FusedBatchNormV3/ReadVariableOp_1?decoder/BN_9/ReadVariableOp?decoder/BN_9/ReadVariableOp_1?1decoder/Conv2DTranspose_10/BiasAdd/ReadVariableOp?:decoder/Conv2DTranspose_10/conv2d_transpose/ReadVariableOp?1decoder/Conv2DTranspose_11/BiasAdd/ReadVariableOp?:decoder/Conv2DTranspose_11/conv2d_transpose/ReadVariableOp?1decoder/Conv2DTranspose_12/BiasAdd/ReadVariableOp?:decoder/Conv2DTranspose_12/conv2d_transpose/ReadVariableOp?1decoder/Conv2DTranspose_13/BiasAdd/ReadVariableOp?:decoder/Conv2DTranspose_13/conv2d_transpose/ReadVariableOp?0decoder/Conv2DTranspose_9/BiasAdd/ReadVariableOp?9decoder/Conv2DTranspose_9/conv2d_transpose/ReadVariableOp?,decoder/Dense_Layer_7/BiasAdd/ReadVariableOp?+decoder/Dense_Layer_7/MatMul/ReadVariableOp?,decoder/Dense_Layer_8/BiasAdd/ReadVariableOp?+decoder/Dense_Layer_8/MatMul/ReadVariableOp?
+decoder/Dense_Layer_7/MatMul/ReadVariableOpReadVariableOp4decoder_dense_layer_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
decoder/Dense_Layer_7/MatMulMatMuldecoder_input3decoder/Dense_Layer_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
,decoder/Dense_Layer_7/BiasAdd/ReadVariableOpReadVariableOp5decoder_dense_layer_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
decoder/Dense_Layer_7/BiasAddBiasAdd&decoder/Dense_Layer_7/MatMul:product:04decoder/Dense_Layer_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????}
decoder/Dense_Layer_7/ReluRelu&decoder/Dense_Layer_7/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
+decoder/Dense_Layer_8/MatMul/ReadVariableOpReadVariableOp4decoder_dense_layer_8_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
decoder/Dense_Layer_8/MatMulMatMul(decoder/Dense_Layer_7/Relu:activations:03decoder/Dense_Layer_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@?
,decoder/Dense_Layer_8/BiasAdd/ReadVariableOpReadVariableOp5decoder_dense_layer_8_biasadd_readvariableop_resource*
_output_shapes	
:?@*
dtype0?
decoder/Dense_Layer_8/BiasAddBiasAdd&decoder/Dense_Layer_8/MatMul:product:04decoder/Dense_Layer_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@}
decoder/Dense_Layer_8/ReluRelu&decoder/Dense_Layer_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????@s
decoder/Reshape_Layer/ShapeShape(decoder/Dense_Layer_8/Relu:activations:0*
T0*
_output_shapes
:s
)decoder/Reshape_Layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+decoder/Reshape_Layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+decoder/Reshape_Layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#decoder/Reshape_Layer/strided_sliceStridedSlice$decoder/Reshape_Layer/Shape:output:02decoder/Reshape_Layer/strided_slice/stack:output:04decoder/Reshape_Layer/strided_slice/stack_1:output:04decoder/Reshape_Layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%decoder/Reshape_Layer/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :g
%decoder/Reshape_Layer/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :h
%decoder/Reshape_Layer/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
#decoder/Reshape_Layer/Reshape/shapePack,decoder/Reshape_Layer/strided_slice:output:0.decoder/Reshape_Layer/Reshape/shape/1:output:0.decoder/Reshape_Layer/Reshape/shape/2:output:0.decoder/Reshape_Layer/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
decoder/Reshape_Layer/ReshapeReshape(decoder/Dense_Layer_8/Relu:activations:0,decoder/Reshape_Layer/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????u
decoder/Conv2DTranspose_9/ShapeShape&decoder/Reshape_Layer/Reshape:output:0*
T0*
_output_shapes
:w
-decoder/Conv2DTranspose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/decoder/Conv2DTranspose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/decoder/Conv2DTranspose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'decoder/Conv2DTranspose_9/strided_sliceStridedSlice(decoder/Conv2DTranspose_9/Shape:output:06decoder/Conv2DTranspose_9/strided_slice/stack:output:08decoder/Conv2DTranspose_9/strided_slice/stack_1:output:08decoder/Conv2DTranspose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!decoder/Conv2DTranspose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :c
!decoder/Conv2DTranspose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
!decoder/Conv2DTranspose_9/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
decoder/Conv2DTranspose_9/stackPack0decoder/Conv2DTranspose_9/strided_slice:output:0*decoder/Conv2DTranspose_9/stack/1:output:0*decoder/Conv2DTranspose_9/stack/2:output:0*decoder/Conv2DTranspose_9/stack/3:output:0*
N*
T0*
_output_shapes
:y
/decoder/Conv2DTranspose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/Conv2DTranspose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/Conv2DTranspose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)decoder/Conv2DTranspose_9/strided_slice_1StridedSlice(decoder/Conv2DTranspose_9/stack:output:08decoder/Conv2DTranspose_9/strided_slice_1/stack:output:0:decoder/Conv2DTranspose_9/strided_slice_1/stack_1:output:0:decoder/Conv2DTranspose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
9decoder/Conv2DTranspose_9/conv2d_transpose/ReadVariableOpReadVariableOpBdecoder_conv2dtranspose_9_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
*decoder/Conv2DTranspose_9/conv2d_transposeConv2DBackpropInput(decoder/Conv2DTranspose_9/stack:output:0Adecoder/Conv2DTranspose_9/conv2d_transpose/ReadVariableOp:value:0&decoder/Reshape_Layer/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
0decoder/Conv2DTranspose_9/BiasAdd/ReadVariableOpReadVariableOp9decoder_conv2dtranspose_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!decoder/Conv2DTranspose_9/BiasAddBiasAdd3decoder/Conv2DTranspose_9/conv2d_transpose:output:08decoder/Conv2DTranspose_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
decoder/Conv2DTranspose_9/ReluRelu*decoder/Conv2DTranspose_9/BiasAdd:output:0*
T0*0
_output_shapes
:??????????}
decoder/BN_9/ReadVariableOpReadVariableOp$decoder_bn_9_readvariableop_resource*
_output_shapes	
:?*
dtype0?
decoder/BN_9/ReadVariableOp_1ReadVariableOp&decoder_bn_9_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
,decoder/BN_9/FusedBatchNormV3/ReadVariableOpReadVariableOp5decoder_bn_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
.decoder/BN_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7decoder_bn_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
decoder/BN_9/FusedBatchNormV3FusedBatchNormV3,decoder/Conv2DTranspose_9/Relu:activations:0#decoder/BN_9/ReadVariableOp:value:0%decoder/BN_9/ReadVariableOp_1:value:04decoder/BN_9/FusedBatchNormV3/ReadVariableOp:value:06decoder/BN_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( q
 decoder/Conv2DTranspose_10/ShapeShape!decoder/BN_9/FusedBatchNormV3:y:0*
T0*
_output_shapes
:x
.decoder/Conv2DTranspose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/Conv2DTranspose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/Conv2DTranspose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/Conv2DTranspose_10/strided_sliceStridedSlice)decoder/Conv2DTranspose_10/Shape:output:07decoder/Conv2DTranspose_10/strided_slice/stack:output:09decoder/Conv2DTranspose_10/strided_slice/stack_1:output:09decoder/Conv2DTranspose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/Conv2DTranspose_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/Conv2DTranspose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :e
"decoder/Conv2DTranspose_10/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
 decoder/Conv2DTranspose_10/stackPack1decoder/Conv2DTranspose_10/strided_slice:output:0+decoder/Conv2DTranspose_10/stack/1:output:0+decoder/Conv2DTranspose_10/stack/2:output:0+decoder/Conv2DTranspose_10/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/Conv2DTranspose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/Conv2DTranspose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/Conv2DTranspose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*decoder/Conv2DTranspose_10/strided_slice_1StridedSlice)decoder/Conv2DTranspose_10/stack:output:09decoder/Conv2DTranspose_10/strided_slice_1/stack:output:0;decoder/Conv2DTranspose_10/strided_slice_1/stack_1:output:0;decoder/Conv2DTranspose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:decoder/Conv2DTranspose_10/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2dtranspose_10_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
+decoder/Conv2DTranspose_10/conv2d_transposeConv2DBackpropInput)decoder/Conv2DTranspose_10/stack:output:0Bdecoder/Conv2DTranspose_10/conv2d_transpose/ReadVariableOp:value:0!decoder/BN_9/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
1decoder/Conv2DTranspose_10/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2dtranspose_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"decoder/Conv2DTranspose_10/BiasAddBiasAdd4decoder/Conv2DTranspose_10/conv2d_transpose:output:09decoder/Conv2DTranspose_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
decoder/Conv2DTranspose_10/ReluRelu+decoder/Conv2DTranspose_10/BiasAdd:output:0*
T0*0
_output_shapes
:??????????
decoder/BN_10/ReadVariableOpReadVariableOp%decoder_bn_10_readvariableop_resource*
_output_shapes	
:?*
dtype0?
decoder/BN_10/ReadVariableOp_1ReadVariableOp'decoder_bn_10_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
-decoder/BN_10/FusedBatchNormV3/ReadVariableOpReadVariableOp6decoder_bn_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
/decoder/BN_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp8decoder_bn_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
decoder/BN_10/FusedBatchNormV3FusedBatchNormV3-decoder/Conv2DTranspose_10/Relu:activations:0$decoder/BN_10/ReadVariableOp:value:0&decoder/BN_10/ReadVariableOp_1:value:05decoder/BN_10/FusedBatchNormV3/ReadVariableOp:value:07decoder/BN_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( r
 decoder/Conv2DTranspose_11/ShapeShape"decoder/BN_10/FusedBatchNormV3:y:0*
T0*
_output_shapes
:x
.decoder/Conv2DTranspose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/Conv2DTranspose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/Conv2DTranspose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/Conv2DTranspose_11/strided_sliceStridedSlice)decoder/Conv2DTranspose_11/Shape:output:07decoder/Conv2DTranspose_11/strided_slice/stack:output:09decoder/Conv2DTranspose_11/strided_slice/stack_1:output:09decoder/Conv2DTranspose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/Conv2DTranspose_11/stack/1Const*
_output_shapes
: *
dtype0*
value	B : d
"decoder/Conv2DTranspose_11/stack/2Const*
_output_shapes
: *
dtype0*
value	B : d
"decoder/Conv2DTranspose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
 decoder/Conv2DTranspose_11/stackPack1decoder/Conv2DTranspose_11/strided_slice:output:0+decoder/Conv2DTranspose_11/stack/1:output:0+decoder/Conv2DTranspose_11/stack/2:output:0+decoder/Conv2DTranspose_11/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/Conv2DTranspose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/Conv2DTranspose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/Conv2DTranspose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*decoder/Conv2DTranspose_11/strided_slice_1StridedSlice)decoder/Conv2DTranspose_11/stack:output:09decoder/Conv2DTranspose_11/strided_slice_1/stack:output:0;decoder/Conv2DTranspose_11/strided_slice_1/stack_1:output:0;decoder/Conv2DTranspose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:decoder/Conv2DTranspose_11/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2dtranspose_11_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
+decoder/Conv2DTranspose_11/conv2d_transposeConv2DBackpropInput)decoder/Conv2DTranspose_11/stack:output:0Bdecoder/Conv2DTranspose_11/conv2d_transpose/ReadVariableOp:value:0"decoder/BN_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
1decoder/Conv2DTranspose_11/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2dtranspose_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
"decoder/Conv2DTranspose_11/BiasAddBiasAdd4decoder/Conv2DTranspose_11/conv2d_transpose:output:09decoder/Conv2DTranspose_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
decoder/Conv2DTranspose_11/ReluRelu+decoder/Conv2DTranspose_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @~
decoder/BN_11/ReadVariableOpReadVariableOp%decoder_bn_11_readvariableop_resource*
_output_shapes
:@*
dtype0?
decoder/BN_11/ReadVariableOp_1ReadVariableOp'decoder_bn_11_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
-decoder/BN_11/FusedBatchNormV3/ReadVariableOpReadVariableOp6decoder_bn_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
/decoder/BN_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp8decoder_bn_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
decoder/BN_11/FusedBatchNormV3FusedBatchNormV3-decoder/Conv2DTranspose_11/Relu:activations:0$decoder/BN_11/ReadVariableOp:value:0&decoder/BN_11/ReadVariableOp_1:value:05decoder/BN_11/FusedBatchNormV3/ReadVariableOp:value:07decoder/BN_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( r
 decoder/Conv2DTranspose_12/ShapeShape"decoder/BN_11/FusedBatchNormV3:y:0*
T0*
_output_shapes
:x
.decoder/Conv2DTranspose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/Conv2DTranspose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/Conv2DTranspose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/Conv2DTranspose_12/strided_sliceStridedSlice)decoder/Conv2DTranspose_12/Shape:output:07decoder/Conv2DTranspose_12/strided_slice/stack:output:09decoder/Conv2DTranspose_12/strided_slice/stack_1:output:09decoder/Conv2DTranspose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/Conv2DTranspose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/Conv2DTranspose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@d
"decoder/Conv2DTranspose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
 decoder/Conv2DTranspose_12/stackPack1decoder/Conv2DTranspose_12/strided_slice:output:0+decoder/Conv2DTranspose_12/stack/1:output:0+decoder/Conv2DTranspose_12/stack/2:output:0+decoder/Conv2DTranspose_12/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/Conv2DTranspose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/Conv2DTranspose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/Conv2DTranspose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*decoder/Conv2DTranspose_12/strided_slice_1StridedSlice)decoder/Conv2DTranspose_12/stack:output:09decoder/Conv2DTranspose_12/strided_slice_1/stack:output:0;decoder/Conv2DTranspose_12/strided_slice_1/stack_1:output:0;decoder/Conv2DTranspose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:decoder/Conv2DTranspose_12/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2dtranspose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
+decoder/Conv2DTranspose_12/conv2d_transposeConv2DBackpropInput)decoder/Conv2DTranspose_12/stack:output:0Bdecoder/Conv2DTranspose_12/conv2d_transpose/ReadVariableOp:value:0"decoder/BN_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
1decoder/Conv2DTranspose_12/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2dtranspose_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
"decoder/Conv2DTranspose_12/BiasAddBiasAdd4decoder/Conv2DTranspose_12/conv2d_transpose:output:09decoder/Conv2DTranspose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ?
decoder/Conv2DTranspose_12/ReluRelu+decoder/Conv2DTranspose_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ ~
decoder/BN_12/ReadVariableOpReadVariableOp%decoder_bn_12_readvariableop_resource*
_output_shapes
: *
dtype0?
decoder/BN_12/ReadVariableOp_1ReadVariableOp'decoder_bn_12_readvariableop_1_resource*
_output_shapes
: *
dtype0?
-decoder/BN_12/FusedBatchNormV3/ReadVariableOpReadVariableOp6decoder_bn_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
/decoder/BN_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp8decoder_bn_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
decoder/BN_12/FusedBatchNormV3FusedBatchNormV3-decoder/Conv2DTranspose_12/Relu:activations:0$decoder/BN_12/ReadVariableOp:value:0&decoder/BN_12/ReadVariableOp_1:value:05decoder/BN_12/FusedBatchNormV3/ReadVariableOp:value:07decoder/BN_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( r
 decoder/Conv2DTranspose_13/ShapeShape"decoder/BN_12/FusedBatchNormV3:y:0*
T0*
_output_shapes
:x
.decoder/Conv2DTranspose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/Conv2DTranspose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/Conv2DTranspose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/Conv2DTranspose_13/strided_sliceStridedSlice)decoder/Conv2DTranspose_13/Shape:output:07decoder/Conv2DTranspose_13/strided_slice/stack:output:09decoder/Conv2DTranspose_13/strided_slice/stack_1:output:09decoder/Conv2DTranspose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"decoder/Conv2DTranspose_13/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?e
"decoder/Conv2DTranspose_13/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?d
"decoder/Conv2DTranspose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
 decoder/Conv2DTranspose_13/stackPack1decoder/Conv2DTranspose_13/strided_slice:output:0+decoder/Conv2DTranspose_13/stack/1:output:0+decoder/Conv2DTranspose_13/stack/2:output:0+decoder/Conv2DTranspose_13/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/Conv2DTranspose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/Conv2DTranspose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/Conv2DTranspose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*decoder/Conv2DTranspose_13/strided_slice_1StridedSlice)decoder/Conv2DTranspose_13/stack:output:09decoder/Conv2DTranspose_13/strided_slice_1/stack:output:0;decoder/Conv2DTranspose_13/strided_slice_1/stack_1:output:0;decoder/Conv2DTranspose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:decoder/Conv2DTranspose_13/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2dtranspose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
+decoder/Conv2DTranspose_13/conv2d_transposeConv2DBackpropInput)decoder/Conv2DTranspose_13/stack:output:0Bdecoder/Conv2DTranspose_13/conv2d_transpose/ReadVariableOp:value:0"decoder/BN_12/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
1decoder/Conv2DTranspose_13/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2dtranspose_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
"decoder/Conv2DTranspose_13/BiasAddBiasAdd4decoder/Conv2DTranspose_13/conv2d_transpose:output:09decoder/Conv2DTranspose_13/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
decoder/Conv2DTranspose_13/TanhTanh+decoder/Conv2DTranspose_13/BiasAdd:output:0*
T0*1
_output_shapes
:???????????|
IdentityIdentity#decoder/Conv2DTranspose_13/Tanh:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp.^decoder/BN_10/FusedBatchNormV3/ReadVariableOp0^decoder/BN_10/FusedBatchNormV3/ReadVariableOp_1^decoder/BN_10/ReadVariableOp^decoder/BN_10/ReadVariableOp_1.^decoder/BN_11/FusedBatchNormV3/ReadVariableOp0^decoder/BN_11/FusedBatchNormV3/ReadVariableOp_1^decoder/BN_11/ReadVariableOp^decoder/BN_11/ReadVariableOp_1.^decoder/BN_12/FusedBatchNormV3/ReadVariableOp0^decoder/BN_12/FusedBatchNormV3/ReadVariableOp_1^decoder/BN_12/ReadVariableOp^decoder/BN_12/ReadVariableOp_1-^decoder/BN_9/FusedBatchNormV3/ReadVariableOp/^decoder/BN_9/FusedBatchNormV3/ReadVariableOp_1^decoder/BN_9/ReadVariableOp^decoder/BN_9/ReadVariableOp_12^decoder/Conv2DTranspose_10/BiasAdd/ReadVariableOp;^decoder/Conv2DTranspose_10/conv2d_transpose/ReadVariableOp2^decoder/Conv2DTranspose_11/BiasAdd/ReadVariableOp;^decoder/Conv2DTranspose_11/conv2d_transpose/ReadVariableOp2^decoder/Conv2DTranspose_12/BiasAdd/ReadVariableOp;^decoder/Conv2DTranspose_12/conv2d_transpose/ReadVariableOp2^decoder/Conv2DTranspose_13/BiasAdd/ReadVariableOp;^decoder/Conv2DTranspose_13/conv2d_transpose/ReadVariableOp1^decoder/Conv2DTranspose_9/BiasAdd/ReadVariableOp:^decoder/Conv2DTranspose_9/conv2d_transpose/ReadVariableOp-^decoder/Dense_Layer_7/BiasAdd/ReadVariableOp,^decoder/Dense_Layer_7/MatMul/ReadVariableOp-^decoder/Dense_Layer_8/BiasAdd/ReadVariableOp,^decoder/Dense_Layer_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-decoder/BN_10/FusedBatchNormV3/ReadVariableOp-decoder/BN_10/FusedBatchNormV3/ReadVariableOp2b
/decoder/BN_10/FusedBatchNormV3/ReadVariableOp_1/decoder/BN_10/FusedBatchNormV3/ReadVariableOp_12<
decoder/BN_10/ReadVariableOpdecoder/BN_10/ReadVariableOp2@
decoder/BN_10/ReadVariableOp_1decoder/BN_10/ReadVariableOp_12^
-decoder/BN_11/FusedBatchNormV3/ReadVariableOp-decoder/BN_11/FusedBatchNormV3/ReadVariableOp2b
/decoder/BN_11/FusedBatchNormV3/ReadVariableOp_1/decoder/BN_11/FusedBatchNormV3/ReadVariableOp_12<
decoder/BN_11/ReadVariableOpdecoder/BN_11/ReadVariableOp2@
decoder/BN_11/ReadVariableOp_1decoder/BN_11/ReadVariableOp_12^
-decoder/BN_12/FusedBatchNormV3/ReadVariableOp-decoder/BN_12/FusedBatchNormV3/ReadVariableOp2b
/decoder/BN_12/FusedBatchNormV3/ReadVariableOp_1/decoder/BN_12/FusedBatchNormV3/ReadVariableOp_12<
decoder/BN_12/ReadVariableOpdecoder/BN_12/ReadVariableOp2@
decoder/BN_12/ReadVariableOp_1decoder/BN_12/ReadVariableOp_12\
,decoder/BN_9/FusedBatchNormV3/ReadVariableOp,decoder/BN_9/FusedBatchNormV3/ReadVariableOp2`
.decoder/BN_9/FusedBatchNormV3/ReadVariableOp_1.decoder/BN_9/FusedBatchNormV3/ReadVariableOp_12:
decoder/BN_9/ReadVariableOpdecoder/BN_9/ReadVariableOp2>
decoder/BN_9/ReadVariableOp_1decoder/BN_9/ReadVariableOp_12f
1decoder/Conv2DTranspose_10/BiasAdd/ReadVariableOp1decoder/Conv2DTranspose_10/BiasAdd/ReadVariableOp2x
:decoder/Conv2DTranspose_10/conv2d_transpose/ReadVariableOp:decoder/Conv2DTranspose_10/conv2d_transpose/ReadVariableOp2f
1decoder/Conv2DTranspose_11/BiasAdd/ReadVariableOp1decoder/Conv2DTranspose_11/BiasAdd/ReadVariableOp2x
:decoder/Conv2DTranspose_11/conv2d_transpose/ReadVariableOp:decoder/Conv2DTranspose_11/conv2d_transpose/ReadVariableOp2f
1decoder/Conv2DTranspose_12/BiasAdd/ReadVariableOp1decoder/Conv2DTranspose_12/BiasAdd/ReadVariableOp2x
:decoder/Conv2DTranspose_12/conv2d_transpose/ReadVariableOp:decoder/Conv2DTranspose_12/conv2d_transpose/ReadVariableOp2f
1decoder/Conv2DTranspose_13/BiasAdd/ReadVariableOp1decoder/Conv2DTranspose_13/BiasAdd/ReadVariableOp2x
:decoder/Conv2DTranspose_13/conv2d_transpose/ReadVariableOp:decoder/Conv2DTranspose_13/conv2d_transpose/ReadVariableOp2d
0decoder/Conv2DTranspose_9/BiasAdd/ReadVariableOp0decoder/Conv2DTranspose_9/BiasAdd/ReadVariableOp2v
9decoder/Conv2DTranspose_9/conv2d_transpose/ReadVariableOp9decoder/Conv2DTranspose_9/conv2d_transpose/ReadVariableOp2\
,decoder/Dense_Layer_7/BiasAdd/ReadVariableOp,decoder/Dense_Layer_7/BiasAdd/ReadVariableOp2Z
+decoder/Dense_Layer_7/MatMul/ReadVariableOp+decoder/Dense_Layer_7/MatMul/ReadVariableOp2\
,decoder/Dense_Layer_8/BiasAdd/ReadVariableOp,decoder/Dense_Layer_8/BiasAdd/ReadVariableOp2Z
+decoder/Dense_Layer_8/MatMul/ReadVariableOp+decoder/Dense_Layer_8/MatMul/ReadVariableOp:W S
(
_output_shapes
:??????????
'
_user_specified_nameDecoder_Input
?
?
3__inference_Conv2DTranspose_12_layer_call_fn_404812

inputs!
unknown: @
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
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
GPU2*0J 8? *W
fRRP
N__inference_Conv2DTranspose_12_layer_call_and_return_conditional_losses_403117?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
@__inference_BN_9_layer_call_and_return_conditional_losses_402850

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_BN_11_layer_call_and_return_conditional_losses_404785

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
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
?
?
2__inference_Conv2DTranspose_9_layer_call_fn_404497

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_Conv2DTranspose_9_layer_call_and_return_conditional_losses_402790?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_BN_9_layer_call_fn_404544

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_BN_9_layer_call_and_return_conditional_losses_402819?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?!
?
M__inference_Conv2DTranspose_9_layer_call_and_return_conditional_losses_404531

inputsD
(conv2d_transpose_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????k
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_decoder_layer_call_fn_403415
decoder_input
unknown:
??
	unknown_0:	?
	unknown_1:
??@
	unknown_2:	?@%
	unknown_3:??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?%

unknown_15:@?

unknown_16:@

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@$

unknown_21: @

unknown_22: 

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: $

unknown_27: 

unknown_28:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldecoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_403352y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameDecoder_Input
?
?
A__inference_BN_12_layer_call_and_return_conditional_losses_404890

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
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
?!
?
N__inference_Conv2DTranspose_13_layer_call_and_return_conditional_losses_403226

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?
C__inference_decoder_layer_call_and_return_conditional_losses_404245

inputs@
,dense_layer_7_matmul_readvariableop_resource:
??<
-dense_layer_7_biasadd_readvariableop_resource:	?@
,dense_layer_8_matmul_readvariableop_resource:
??@<
-dense_layer_8_biasadd_readvariableop_resource:	?@V
:conv2dtranspose_9_conv2d_transpose_readvariableop_resource:??@
1conv2dtranspose_9_biasadd_readvariableop_resource:	?+
bn_9_readvariableop_resource:	?-
bn_9_readvariableop_1_resource:	?<
-bn_9_fusedbatchnormv3_readvariableop_resource:	?>
/bn_9_fusedbatchnormv3_readvariableop_1_resource:	?W
;conv2dtranspose_10_conv2d_transpose_readvariableop_resource:??A
2conv2dtranspose_10_biasadd_readvariableop_resource:	?,
bn_10_readvariableop_resource:	?.
bn_10_readvariableop_1_resource:	?=
.bn_10_fusedbatchnormv3_readvariableop_resource:	??
0bn_10_fusedbatchnormv3_readvariableop_1_resource:	?V
;conv2dtranspose_11_conv2d_transpose_readvariableop_resource:@?@
2conv2dtranspose_11_biasadd_readvariableop_resource:@+
bn_11_readvariableop_resource:@-
bn_11_readvariableop_1_resource:@<
.bn_11_fusedbatchnormv3_readvariableop_resource:@>
0bn_11_fusedbatchnormv3_readvariableop_1_resource:@U
;conv2dtranspose_12_conv2d_transpose_readvariableop_resource: @@
2conv2dtranspose_12_biasadd_readvariableop_resource: +
bn_12_readvariableop_resource: -
bn_12_readvariableop_1_resource: <
.bn_12_fusedbatchnormv3_readvariableop_resource: >
0bn_12_fusedbatchnormv3_readvariableop_1_resource: U
;conv2dtranspose_13_conv2d_transpose_readvariableop_resource: @
2conv2dtranspose_13_biasadd_readvariableop_resource:
identity??%BN_10/FusedBatchNormV3/ReadVariableOp?'BN_10/FusedBatchNormV3/ReadVariableOp_1?BN_10/ReadVariableOp?BN_10/ReadVariableOp_1?%BN_11/FusedBatchNormV3/ReadVariableOp?'BN_11/FusedBatchNormV3/ReadVariableOp_1?BN_11/ReadVariableOp?BN_11/ReadVariableOp_1?%BN_12/FusedBatchNormV3/ReadVariableOp?'BN_12/FusedBatchNormV3/ReadVariableOp_1?BN_12/ReadVariableOp?BN_12/ReadVariableOp_1?$BN_9/FusedBatchNormV3/ReadVariableOp?&BN_9/FusedBatchNormV3/ReadVariableOp_1?BN_9/ReadVariableOp?BN_9/ReadVariableOp_1?)Conv2DTranspose_10/BiasAdd/ReadVariableOp?2Conv2DTranspose_10/conv2d_transpose/ReadVariableOp?)Conv2DTranspose_11/BiasAdd/ReadVariableOp?2Conv2DTranspose_11/conv2d_transpose/ReadVariableOp?)Conv2DTranspose_12/BiasAdd/ReadVariableOp?2Conv2DTranspose_12/conv2d_transpose/ReadVariableOp?)Conv2DTranspose_13/BiasAdd/ReadVariableOp?2Conv2DTranspose_13/conv2d_transpose/ReadVariableOp?(Conv2DTranspose_9/BiasAdd/ReadVariableOp?1Conv2DTranspose_9/conv2d_transpose/ReadVariableOp?$Dense_Layer_7/BiasAdd/ReadVariableOp?#Dense_Layer_7/MatMul/ReadVariableOp?$Dense_Layer_8/BiasAdd/ReadVariableOp?#Dense_Layer_8/MatMul/ReadVariableOp?
#Dense_Layer_7/MatMul/ReadVariableOpReadVariableOp,dense_layer_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Dense_Layer_7/MatMulMatMulinputs+Dense_Layer_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$Dense_Layer_7/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Dense_Layer_7/BiasAddBiasAddDense_Layer_7/MatMul:product:0,Dense_Layer_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????m
Dense_Layer_7/ReluReluDense_Layer_7/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
#Dense_Layer_8/MatMul/ReadVariableOpReadVariableOp,dense_layer_8_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
Dense_Layer_8/MatMulMatMul Dense_Layer_7/Relu:activations:0+Dense_Layer_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@?
$Dense_Layer_8/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_8_biasadd_readvariableop_resource*
_output_shapes	
:?@*
dtype0?
Dense_Layer_8/BiasAddBiasAddDense_Layer_8/MatMul:product:0,Dense_Layer_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@m
Dense_Layer_8/ReluReluDense_Layer_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????@c
Reshape_Layer/ShapeShape Dense_Layer_8/Relu:activations:0*
T0*
_output_shapes
:k
!Reshape_Layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Reshape_Layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Reshape_Layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Reshape_Layer/strided_sliceStridedSliceReshape_Layer/Shape:output:0*Reshape_Layer/strided_slice/stack:output:0,Reshape_Layer/strided_slice/stack_1:output:0,Reshape_Layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
Reshape_Layer/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :_
Reshape_Layer/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`
Reshape_Layer/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
Reshape_Layer/Reshape/shapePack$Reshape_Layer/strided_slice:output:0&Reshape_Layer/Reshape/shape/1:output:0&Reshape_Layer/Reshape/shape/2:output:0&Reshape_Layer/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
Reshape_Layer/ReshapeReshape Dense_Layer_8/Relu:activations:0$Reshape_Layer/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????e
Conv2DTranspose_9/ShapeShapeReshape_Layer/Reshape:output:0*
T0*
_output_shapes
:o
%Conv2DTranspose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'Conv2DTranspose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'Conv2DTranspose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Conv2DTranspose_9/strided_sliceStridedSlice Conv2DTranspose_9/Shape:output:0.Conv2DTranspose_9/strided_slice/stack:output:00Conv2DTranspose_9/strided_slice/stack_1:output:00Conv2DTranspose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
Conv2DTranspose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :[
Conv2DTranspose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
Conv2DTranspose_9/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
Conv2DTranspose_9/stackPack(Conv2DTranspose_9/strided_slice:output:0"Conv2DTranspose_9/stack/1:output:0"Conv2DTranspose_9/stack/2:output:0"Conv2DTranspose_9/stack/3:output:0*
N*
T0*
_output_shapes
:q
'Conv2DTranspose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)Conv2DTranspose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)Conv2DTranspose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!Conv2DTranspose_9/strided_slice_1StridedSlice Conv2DTranspose_9/stack:output:00Conv2DTranspose_9/strided_slice_1/stack:output:02Conv2DTranspose_9/strided_slice_1/stack_1:output:02Conv2DTranspose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
1Conv2DTranspose_9/conv2d_transpose/ReadVariableOpReadVariableOp:conv2dtranspose_9_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
"Conv2DTranspose_9/conv2d_transposeConv2DBackpropInput Conv2DTranspose_9/stack:output:09Conv2DTranspose_9/conv2d_transpose/ReadVariableOp:value:0Reshape_Layer/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
(Conv2DTranspose_9/BiasAdd/ReadVariableOpReadVariableOp1conv2dtranspose_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Conv2DTranspose_9/BiasAddBiasAdd+Conv2DTranspose_9/conv2d_transpose:output:00Conv2DTranspose_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????}
Conv2DTranspose_9/ReluRelu"Conv2DTranspose_9/BiasAdd:output:0*
T0*0
_output_shapes
:??????????m
BN_9/ReadVariableOpReadVariableOpbn_9_readvariableop_resource*
_output_shapes	
:?*
dtype0q
BN_9/ReadVariableOp_1ReadVariableOpbn_9_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
$BN_9/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&BN_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
BN_9/FusedBatchNormV3FusedBatchNormV3$Conv2DTranspose_9/Relu:activations:0BN_9/ReadVariableOp:value:0BN_9/ReadVariableOp_1:value:0,BN_9/FusedBatchNormV3/ReadVariableOp:value:0.BN_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( a
Conv2DTranspose_10/ShapeShapeBN_9/FusedBatchNormV3:y:0*
T0*
_output_shapes
:p
&Conv2DTranspose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(Conv2DTranspose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(Conv2DTranspose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 Conv2DTranspose_10/strided_sliceStridedSlice!Conv2DTranspose_10/Shape:output:0/Conv2DTranspose_10/strided_slice/stack:output:01Conv2DTranspose_10/strided_slice/stack_1:output:01Conv2DTranspose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
Conv2DTranspose_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
Conv2DTranspose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
Conv2DTranspose_10/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
Conv2DTranspose_10/stackPack)Conv2DTranspose_10/strided_slice:output:0#Conv2DTranspose_10/stack/1:output:0#Conv2DTranspose_10/stack/2:output:0#Conv2DTranspose_10/stack/3:output:0*
N*
T0*
_output_shapes
:r
(Conv2DTranspose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Conv2DTranspose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Conv2DTranspose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Conv2DTranspose_10/strided_slice_1StridedSlice!Conv2DTranspose_10/stack:output:01Conv2DTranspose_10/strided_slice_1/stack:output:03Conv2DTranspose_10/strided_slice_1/stack_1:output:03Conv2DTranspose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2Conv2DTranspose_10/conv2d_transpose/ReadVariableOpReadVariableOp;conv2dtranspose_10_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
#Conv2DTranspose_10/conv2d_transposeConv2DBackpropInput!Conv2DTranspose_10/stack:output:0:Conv2DTranspose_10/conv2d_transpose/ReadVariableOp:value:0BN_9/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)Conv2DTranspose_10/BiasAdd/ReadVariableOpReadVariableOp2conv2dtranspose_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Conv2DTranspose_10/BiasAddBiasAdd,Conv2DTranspose_10/conv2d_transpose:output:01Conv2DTranspose_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
Conv2DTranspose_10/ReluRelu#Conv2DTranspose_10/BiasAdd:output:0*
T0*0
_output_shapes
:??????????o
BN_10/ReadVariableOpReadVariableOpbn_10_readvariableop_resource*
_output_shapes	
:?*
dtype0s
BN_10/ReadVariableOp_1ReadVariableOpbn_10_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
%BN_10/FusedBatchNormV3/ReadVariableOpReadVariableOp.bn_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'BN_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0bn_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
BN_10/FusedBatchNormV3FusedBatchNormV3%Conv2DTranspose_10/Relu:activations:0BN_10/ReadVariableOp:value:0BN_10/ReadVariableOp_1:value:0-BN_10/FusedBatchNormV3/ReadVariableOp:value:0/BN_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( b
Conv2DTranspose_11/ShapeShapeBN_10/FusedBatchNormV3:y:0*
T0*
_output_shapes
:p
&Conv2DTranspose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(Conv2DTranspose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(Conv2DTranspose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 Conv2DTranspose_11/strided_sliceStridedSlice!Conv2DTranspose_11/Shape:output:0/Conv2DTranspose_11/strided_slice/stack:output:01Conv2DTranspose_11/strided_slice/stack_1:output:01Conv2DTranspose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
Conv2DTranspose_11/stack/1Const*
_output_shapes
: *
dtype0*
value	B : \
Conv2DTranspose_11/stack/2Const*
_output_shapes
: *
dtype0*
value	B : \
Conv2DTranspose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
Conv2DTranspose_11/stackPack)Conv2DTranspose_11/strided_slice:output:0#Conv2DTranspose_11/stack/1:output:0#Conv2DTranspose_11/stack/2:output:0#Conv2DTranspose_11/stack/3:output:0*
N*
T0*
_output_shapes
:r
(Conv2DTranspose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Conv2DTranspose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Conv2DTranspose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Conv2DTranspose_11/strided_slice_1StridedSlice!Conv2DTranspose_11/stack:output:01Conv2DTranspose_11/strided_slice_1/stack:output:03Conv2DTranspose_11/strided_slice_1/stack_1:output:03Conv2DTranspose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2Conv2DTranspose_11/conv2d_transpose/ReadVariableOpReadVariableOp;conv2dtranspose_11_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
#Conv2DTranspose_11/conv2d_transposeConv2DBackpropInput!Conv2DTranspose_11/stack:output:0:Conv2DTranspose_11/conv2d_transpose/ReadVariableOp:value:0BN_10/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
)Conv2DTranspose_11/BiasAdd/ReadVariableOpReadVariableOp2conv2dtranspose_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Conv2DTranspose_11/BiasAddBiasAdd,Conv2DTranspose_11/conv2d_transpose:output:01Conv2DTranspose_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @~
Conv2DTranspose_11/ReluRelu#Conv2DTranspose_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @n
BN_11/ReadVariableOpReadVariableOpbn_11_readvariableop_resource*
_output_shapes
:@*
dtype0r
BN_11/ReadVariableOp_1ReadVariableOpbn_11_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
%BN_11/FusedBatchNormV3/ReadVariableOpReadVariableOp.bn_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
'BN_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0bn_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
BN_11/FusedBatchNormV3FusedBatchNormV3%Conv2DTranspose_11/Relu:activations:0BN_11/ReadVariableOp:value:0BN_11/ReadVariableOp_1:value:0-BN_11/FusedBatchNormV3/ReadVariableOp:value:0/BN_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( b
Conv2DTranspose_12/ShapeShapeBN_11/FusedBatchNormV3:y:0*
T0*
_output_shapes
:p
&Conv2DTranspose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(Conv2DTranspose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(Conv2DTranspose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 Conv2DTranspose_12/strided_sliceStridedSlice!Conv2DTranspose_12/Shape:output:0/Conv2DTranspose_12/strided_slice/stack:output:01Conv2DTranspose_12/strided_slice/stack_1:output:01Conv2DTranspose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
Conv2DTranspose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
Conv2DTranspose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
Conv2DTranspose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
Conv2DTranspose_12/stackPack)Conv2DTranspose_12/strided_slice:output:0#Conv2DTranspose_12/stack/1:output:0#Conv2DTranspose_12/stack/2:output:0#Conv2DTranspose_12/stack/3:output:0*
N*
T0*
_output_shapes
:r
(Conv2DTranspose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Conv2DTranspose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Conv2DTranspose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Conv2DTranspose_12/strided_slice_1StridedSlice!Conv2DTranspose_12/stack:output:01Conv2DTranspose_12/strided_slice_1/stack:output:03Conv2DTranspose_12/strided_slice_1/stack_1:output:03Conv2DTranspose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2Conv2DTranspose_12/conv2d_transpose/ReadVariableOpReadVariableOp;conv2dtranspose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
#Conv2DTranspose_12/conv2d_transposeConv2DBackpropInput!Conv2DTranspose_12/stack:output:0:Conv2DTranspose_12/conv2d_transpose/ReadVariableOp:value:0BN_11/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
?
)Conv2DTranspose_12/BiasAdd/ReadVariableOpReadVariableOp2conv2dtranspose_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Conv2DTranspose_12/BiasAddBiasAdd,Conv2DTranspose_12/conv2d_transpose:output:01Conv2DTranspose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ ~
Conv2DTranspose_12/ReluRelu#Conv2DTranspose_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ n
BN_12/ReadVariableOpReadVariableOpbn_12_readvariableop_resource*
_output_shapes
: *
dtype0r
BN_12/ReadVariableOp_1ReadVariableOpbn_12_readvariableop_1_resource*
_output_shapes
: *
dtype0?
%BN_12/FusedBatchNormV3/ReadVariableOpReadVariableOp.bn_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
'BN_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp0bn_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
BN_12/FusedBatchNormV3FusedBatchNormV3%Conv2DTranspose_12/Relu:activations:0BN_12/ReadVariableOp:value:0BN_12/ReadVariableOp_1:value:0-BN_12/FusedBatchNormV3/ReadVariableOp:value:0/BN_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( b
Conv2DTranspose_13/ShapeShapeBN_12/FusedBatchNormV3:y:0*
T0*
_output_shapes
:p
&Conv2DTranspose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(Conv2DTranspose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(Conv2DTranspose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 Conv2DTranspose_13/strided_sliceStridedSlice!Conv2DTranspose_13/Shape:output:0/Conv2DTranspose_13/strided_slice/stack:output:01Conv2DTranspose_13/strided_slice/stack_1:output:01Conv2DTranspose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
Conv2DTranspose_13/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?]
Conv2DTranspose_13/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?\
Conv2DTranspose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
Conv2DTranspose_13/stackPack)Conv2DTranspose_13/strided_slice:output:0#Conv2DTranspose_13/stack/1:output:0#Conv2DTranspose_13/stack/2:output:0#Conv2DTranspose_13/stack/3:output:0*
N*
T0*
_output_shapes
:r
(Conv2DTranspose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Conv2DTranspose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Conv2DTranspose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"Conv2DTranspose_13/strided_slice_1StridedSlice!Conv2DTranspose_13/stack:output:01Conv2DTranspose_13/strided_slice_1/stack:output:03Conv2DTranspose_13/strided_slice_1/stack_1:output:03Conv2DTranspose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2Conv2DTranspose_13/conv2d_transpose/ReadVariableOpReadVariableOp;conv2dtranspose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
#Conv2DTranspose_13/conv2d_transposeConv2DBackpropInput!Conv2DTranspose_13/stack:output:0:Conv2DTranspose_13/conv2d_transpose/ReadVariableOp:value:0BN_12/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
)Conv2DTranspose_13/BiasAdd/ReadVariableOpReadVariableOp2conv2dtranspose_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Conv2DTranspose_13/BiasAddBiasAdd,Conv2DTranspose_13/conv2d_transpose:output:01Conv2DTranspose_13/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
Conv2DTranspose_13/TanhTanh#Conv2DTranspose_13/BiasAdd:output:0*
T0*1
_output_shapes
:???????????t
IdentityIdentityConv2DTranspose_13/Tanh:y:0^NoOp*
T0*1
_output_shapes
:????????????	
NoOpNoOp&^BN_10/FusedBatchNormV3/ReadVariableOp(^BN_10/FusedBatchNormV3/ReadVariableOp_1^BN_10/ReadVariableOp^BN_10/ReadVariableOp_1&^BN_11/FusedBatchNormV3/ReadVariableOp(^BN_11/FusedBatchNormV3/ReadVariableOp_1^BN_11/ReadVariableOp^BN_11/ReadVariableOp_1&^BN_12/FusedBatchNormV3/ReadVariableOp(^BN_12/FusedBatchNormV3/ReadVariableOp_1^BN_12/ReadVariableOp^BN_12/ReadVariableOp_1%^BN_9/FusedBatchNormV3/ReadVariableOp'^BN_9/FusedBatchNormV3/ReadVariableOp_1^BN_9/ReadVariableOp^BN_9/ReadVariableOp_1*^Conv2DTranspose_10/BiasAdd/ReadVariableOp3^Conv2DTranspose_10/conv2d_transpose/ReadVariableOp*^Conv2DTranspose_11/BiasAdd/ReadVariableOp3^Conv2DTranspose_11/conv2d_transpose/ReadVariableOp*^Conv2DTranspose_12/BiasAdd/ReadVariableOp3^Conv2DTranspose_12/conv2d_transpose/ReadVariableOp*^Conv2DTranspose_13/BiasAdd/ReadVariableOp3^Conv2DTranspose_13/conv2d_transpose/ReadVariableOp)^Conv2DTranspose_9/BiasAdd/ReadVariableOp2^Conv2DTranspose_9/conv2d_transpose/ReadVariableOp%^Dense_Layer_7/BiasAdd/ReadVariableOp$^Dense_Layer_7/MatMul/ReadVariableOp%^Dense_Layer_8/BiasAdd/ReadVariableOp$^Dense_Layer_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%BN_10/FusedBatchNormV3/ReadVariableOp%BN_10/FusedBatchNormV3/ReadVariableOp2R
'BN_10/FusedBatchNormV3/ReadVariableOp_1'BN_10/FusedBatchNormV3/ReadVariableOp_12,
BN_10/ReadVariableOpBN_10/ReadVariableOp20
BN_10/ReadVariableOp_1BN_10/ReadVariableOp_12N
%BN_11/FusedBatchNormV3/ReadVariableOp%BN_11/FusedBatchNormV3/ReadVariableOp2R
'BN_11/FusedBatchNormV3/ReadVariableOp_1'BN_11/FusedBatchNormV3/ReadVariableOp_12,
BN_11/ReadVariableOpBN_11/ReadVariableOp20
BN_11/ReadVariableOp_1BN_11/ReadVariableOp_12N
%BN_12/FusedBatchNormV3/ReadVariableOp%BN_12/FusedBatchNormV3/ReadVariableOp2R
'BN_12/FusedBatchNormV3/ReadVariableOp_1'BN_12/FusedBatchNormV3/ReadVariableOp_12,
BN_12/ReadVariableOpBN_12/ReadVariableOp20
BN_12/ReadVariableOp_1BN_12/ReadVariableOp_12L
$BN_9/FusedBatchNormV3/ReadVariableOp$BN_9/FusedBatchNormV3/ReadVariableOp2P
&BN_9/FusedBatchNormV3/ReadVariableOp_1&BN_9/FusedBatchNormV3/ReadVariableOp_12*
BN_9/ReadVariableOpBN_9/ReadVariableOp2.
BN_9/ReadVariableOp_1BN_9/ReadVariableOp_12V
)Conv2DTranspose_10/BiasAdd/ReadVariableOp)Conv2DTranspose_10/BiasAdd/ReadVariableOp2h
2Conv2DTranspose_10/conv2d_transpose/ReadVariableOp2Conv2DTranspose_10/conv2d_transpose/ReadVariableOp2V
)Conv2DTranspose_11/BiasAdd/ReadVariableOp)Conv2DTranspose_11/BiasAdd/ReadVariableOp2h
2Conv2DTranspose_11/conv2d_transpose/ReadVariableOp2Conv2DTranspose_11/conv2d_transpose/ReadVariableOp2V
)Conv2DTranspose_12/BiasAdd/ReadVariableOp)Conv2DTranspose_12/BiasAdd/ReadVariableOp2h
2Conv2DTranspose_12/conv2d_transpose/ReadVariableOp2Conv2DTranspose_12/conv2d_transpose/ReadVariableOp2V
)Conv2DTranspose_13/BiasAdd/ReadVariableOp)Conv2DTranspose_13/BiasAdd/ReadVariableOp2h
2Conv2DTranspose_13/conv2d_transpose/ReadVariableOp2Conv2DTranspose_13/conv2d_transpose/ReadVariableOp2T
(Conv2DTranspose_9/BiasAdd/ReadVariableOp(Conv2DTranspose_9/BiasAdd/ReadVariableOp2f
1Conv2DTranspose_9/conv2d_transpose/ReadVariableOp1Conv2DTranspose_9/conv2d_transpose/ReadVariableOp2L
$Dense_Layer_7/BiasAdd/ReadVariableOp$Dense_Layer_7/BiasAdd/ReadVariableOp2J
#Dense_Layer_7/MatMul/ReadVariableOp#Dense_Layer_7/MatMul/ReadVariableOp2L
$Dense_Layer_8/BiasAdd/ReadVariableOp$Dense_Layer_8/BiasAdd/ReadVariableOp2J
#Dense_Layer_8/MatMul/ReadVariableOp#Dense_Layer_8/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_BN_9_layer_call_and_return_conditional_losses_404575

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_BN_12_layer_call_and_return_conditional_losses_403177

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
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
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
H
Decoder_Input7
serving_default_Decoder_Input:0??????????P
Conv2DTranspose_13:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
?
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias
 4_jit_compiled_convolution_op"
_tf_keras_layer
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;axis
	<gamma
=beta
>moving_mean
?moving_variance"
_tf_keras_layer
?
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias
 H_jit_compiled_convolution_op"
_tf_keras_layer
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance"
_tf_keras_layer
?
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

Zkernel
[bias
 \_jit_compiled_convolution_op"
_tf_keras_layer
?
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses
caxis
	dgamma
ebeta
fmoving_mean
gmoving_variance"
_tf_keras_layer
?
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses

nkernel
obias
 p_jit_compiled_convolution_op"
_tf_keras_layer
?
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance"
_tf_keras_layer
?
|	variables
}trainable_variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
0
1
$2
%3
24
35
<6
=7
>8
?9
F10
G11
P12
Q13
R14
S15
Z16
[17
d18
e19
f20
g21
n22
o23
x24
y25
z26
{27
?28
?29"
trackable_list_wrapper
?
0
1
$2
%3
24
35
<6
=7
F8
G9
P10
Q11
Z12
[13
d14
e15
n16
o17
x18
y19
?20
?21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
(__inference_decoder_layer_call_fn_403415
(__inference_decoder_layer_call_fn_403996
(__inference_decoder_layer_call_fn_404061
(__inference_decoder_layer_call_fn_403712?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
C__inference_decoder_layer_call_and_return_conditional_losses_404245
C__inference_decoder_layer_call_and_return_conditional_losses_404429
C__inference_decoder_layer_call_and_return_conditional_losses_403788
C__inference_decoder_layer_call_and_return_conditional_losses_403864?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?B?
!__inference__wrapped_model_402752Decoder_Input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
-
?serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
.__inference_Dense_Layer_7_layer_call_fn_404438?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
I__inference_Dense_Layer_7_layer_call_and_return_conditional_losses_404449?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
(:&
??2Dense_Layer_7/kernel
!:?2Dense_Layer_7/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
.__inference_Dense_Layer_8_layer_call_fn_404458?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
I__inference_Dense_Layer_8_layer_call_and_return_conditional_losses_404469?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
(:&
??@2Dense_Layer_8/kernel
!:?@2Dense_Layer_8/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
.__inference_Reshape_Layer_layer_call_fn_404474?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
I__inference_Reshape_Layer_layer_call_and_return_conditional_losses_404488?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
2__inference_Conv2DTranspose_9_layer_call_fn_404497?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
M__inference_Conv2DTranspose_9_layer_call_and_return_conditional_losses_404531?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
4:2??2Conv2DTranspose_9/kernel
%:#?2Conv2DTranspose_9/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
<
<0
=1
>2
?3"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
%__inference_BN_9_layer_call_fn_404544
%__inference_BN_9_layer_call_fn_404557?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
@__inference_BN_9_layer_call_and_return_conditional_losses_404575
@__inference_BN_9_layer_call_and_return_conditional_losses_404593?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
:?2
BN_9/gamma
:?2	BN_9/beta
!:? (2BN_9/moving_mean
%:#? (2BN_9/moving_variance
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
3__inference_Conv2DTranspose_10_layer_call_fn_404602?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
N__inference_Conv2DTranspose_10_layer_call_and_return_conditional_losses_404636?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
5:3??2Conv2DTranspose_10/kernel
&:$?2Conv2DTranspose_10/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
<
P0
Q1
R2
S3"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
&__inference_BN_10_layer_call_fn_404649
&__inference_BN_10_layer_call_fn_404662?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
A__inference_BN_10_layer_call_and_return_conditional_losses_404680
A__inference_BN_10_layer_call_and_return_conditional_losses_404698?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
:?2BN_10/gamma
:?2
BN_10/beta
": ? (2BN_10/moving_mean
&:$? (2BN_10/moving_variance
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
3__inference_Conv2DTranspose_11_layer_call_fn_404707?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
N__inference_Conv2DTranspose_11_layer_call_and_return_conditional_losses_404741?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
4:2@?2Conv2DTranspose_11/kernel
%:#@2Conv2DTranspose_11/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
<
d0
e1
f2
g3"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
&__inference_BN_11_layer_call_fn_404754
&__inference_BN_11_layer_call_fn_404767?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
A__inference_BN_11_layer_call_and_return_conditional_losses_404785
A__inference_BN_11_layer_call_and_return_conditional_losses_404803?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
:@2BN_11/gamma
:@2
BN_11/beta
!:@ (2BN_11/moving_mean
%:#@ (2BN_11/moving_variance
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
3__inference_Conv2DTranspose_12_layer_call_fn_404812?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
N__inference_Conv2DTranspose_12_layer_call_and_return_conditional_losses_404846?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
3:1 @2Conv2DTranspose_12/kernel
%:# 2Conv2DTranspose_12/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
<
x0
y1
z2
{3"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
&__inference_BN_12_layer_call_fn_404859
&__inference_BN_12_layer_call_fn_404872?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
A__inference_BN_12_layer_call_and_return_conditional_losses_404890
A__inference_BN_12_layer_call_and_return_conditional_losses_404908?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
: 2BN_12/gamma
: 2
BN_12/beta
!:  (2BN_12/moving_mean
%:#  (2BN_12/moving_variance
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
3__inference_Conv2DTranspose_13_layer_call_fn_404917?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
N__inference_Conv2DTranspose_13_layer_call_and_return_conditional_losses_404951?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
3:1 2Conv2DTranspose_13/kernel
%:#2Conv2DTranspose_13/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
X
>0
?1
R2
S3
f4
g5
z6
{7"
trackable_list_wrapper
~
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
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_decoder_layer_call_fn_403415Decoder_Input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
(__inference_decoder_layer_call_fn_403996inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
(__inference_decoder_layer_call_fn_404061inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
(__inference_decoder_layer_call_fn_403712Decoder_Input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_decoder_layer_call_and_return_conditional_losses_404245inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_decoder_layer_call_and_return_conditional_losses_404429inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_decoder_layer_call_and_return_conditional_losses_403788Decoder_Input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_decoder_layer_call_and_return_conditional_losses_403864Decoder_Input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
$__inference_signature_wrapper_403931Decoder_Input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
.__inference_Dense_Layer_7_layer_call_fn_404438inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_Dense_Layer_7_layer_call_and_return_conditional_losses_404449inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
.__inference_Dense_Layer_8_layer_call_fn_404458inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_Dense_Layer_8_layer_call_and_return_conditional_losses_404469inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
.__inference_Reshape_Layer_layer_call_fn_404474inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_Reshape_Layer_layer_call_and_return_conditional_losses_404488inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
2__inference_Conv2DTranspose_9_layer_call_fn_404497inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
M__inference_Conv2DTranspose_9_layer_call_and_return_conditional_losses_404531inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_BN_9_layer_call_fn_404544inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_BN_9_layer_call_fn_404557inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_BN_9_layer_call_and_return_conditional_losses_404575inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_BN_9_layer_call_and_return_conditional_losses_404593inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
3__inference_Conv2DTranspose_10_layer_call_fn_404602inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_Conv2DTranspose_10_layer_call_and_return_conditional_losses_404636inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
&__inference_BN_10_layer_call_fn_404649inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
&__inference_BN_10_layer_call_fn_404662inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
A__inference_BN_10_layer_call_and_return_conditional_losses_404680inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
A__inference_BN_10_layer_call_and_return_conditional_losses_404698inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
3__inference_Conv2DTranspose_11_layer_call_fn_404707inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_Conv2DTranspose_11_layer_call_and_return_conditional_losses_404741inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
&__inference_BN_11_layer_call_fn_404754inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
&__inference_BN_11_layer_call_fn_404767inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
A__inference_BN_11_layer_call_and_return_conditional_losses_404785inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
A__inference_BN_11_layer_call_and_return_conditional_losses_404803inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
3__inference_Conv2DTranspose_12_layer_call_fn_404812inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_Conv2DTranspose_12_layer_call_and_return_conditional_losses_404846inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
&__inference_BN_12_layer_call_fn_404859inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
&__inference_BN_12_layer_call_fn_404872inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
A__inference_BN_12_layer_call_and_return_conditional_losses_404890inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
A__inference_BN_12_layer_call_and_return_conditional_losses_404908inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
3__inference_Conv2DTranspose_13_layer_call_fn_404917inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_Conv2DTranspose_13_layer_call_and_return_conditional_losses_404951inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
A__inference_BN_10_layer_call_and_return_conditional_losses_404680?PQRSN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
A__inference_BN_10_layer_call_and_return_conditional_losses_404698?PQRSN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
&__inference_BN_10_layer_call_fn_404649?PQRSN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
&__inference_BN_10_layer_call_fn_404662?PQRSN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
A__inference_BN_11_layer_call_and_return_conditional_losses_404785?defgM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
A__inference_BN_11_layer_call_and_return_conditional_losses_404803?defgM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
&__inference_BN_11_layer_call_fn_404754?defgM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
&__inference_BN_11_layer_call_fn_404767?defgM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
A__inference_BN_12_layer_call_and_return_conditional_losses_404890?xyz{M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
A__inference_BN_12_layer_call_and_return_conditional_losses_404908?xyz{M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
&__inference_BN_12_layer_call_fn_404859?xyz{M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
&__inference_BN_12_layer_call_fn_404872?xyz{M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
@__inference_BN_9_layer_call_and_return_conditional_losses_404575?<=>?N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
@__inference_BN_9_layer_call_and_return_conditional_losses_404593?<=>?N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
%__inference_BN_9_layer_call_fn_404544?<=>?N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
%__inference_BN_9_layer_call_fn_404557?<=>?N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
N__inference_Conv2DTranspose_10_layer_call_and_return_conditional_losses_404636?FGJ?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
3__inference_Conv2DTranspose_10_layer_call_fn_404602?FGJ?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
N__inference_Conv2DTranspose_11_layer_call_and_return_conditional_losses_404741?Z[J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
3__inference_Conv2DTranspose_11_layer_call_fn_404707?Z[J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
N__inference_Conv2DTranspose_12_layer_call_and_return_conditional_losses_404846?noI?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
3__inference_Conv2DTranspose_12_layer_call_fn_404812?noI?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
N__inference_Conv2DTranspose_13_layer_call_and_return_conditional_losses_404951???I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
3__inference_Conv2DTranspose_13_layer_call_fn_404917???I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
M__inference_Conv2DTranspose_9_layer_call_and_return_conditional_losses_404531?23J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
2__inference_Conv2DTranspose_9_layer_call_fn_404497?23J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
I__inference_Dense_Layer_7_layer_call_and_return_conditional_losses_404449^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
.__inference_Dense_Layer_7_layer_call_fn_404438Q0?-
&?#
!?
inputs??????????
? "????????????
I__inference_Dense_Layer_8_layer_call_and_return_conditional_losses_404469^$%0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????@
? ?
.__inference_Dense_Layer_8_layer_call_fn_404458Q$%0?-
&?#
!?
inputs??????????
? "???????????@?
I__inference_Reshape_Layer_layer_call_and_return_conditional_losses_404488b0?-
&?#
!?
inputs??????????@
? ".?+
$?!
0??????????
? ?
.__inference_Reshape_Layer_layer_call_fn_404474U0?-
&?#
!?
inputs??????????@
? "!????????????
!__inference__wrapped_model_402752? $%23<=>?FGPQRSZ[defgnoxyz{??7?4
-?*
(?%
Decoder_Input??????????
? "Q?N
L
Conv2DTranspose_136?3
Conv2DTranspose_13????????????
C__inference_decoder_layer_call_and_return_conditional_losses_403788? $%23<=>?FGPQRSZ[defgnoxyz{????<
5?2
(?%
Decoder_Input??????????
p 

 
? "/?,
%?"
0???????????
? ?
C__inference_decoder_layer_call_and_return_conditional_losses_403864? $%23<=>?FGPQRSZ[defgnoxyz{????<
5?2
(?%
Decoder_Input??????????
p

 
? "/?,
%?"
0???????????
? ?
C__inference_decoder_layer_call_and_return_conditional_losses_404245? $%23<=>?FGPQRSZ[defgnoxyz{??8?5
.?+
!?
inputs??????????
p 

 
? "/?,
%?"
0???????????
? ?
C__inference_decoder_layer_call_and_return_conditional_losses_404429? $%23<=>?FGPQRSZ[defgnoxyz{??8?5
.?+
!?
inputs??????????
p

 
? "/?,
%?"
0???????????
? ?
(__inference_decoder_layer_call_fn_403415? $%23<=>?FGPQRSZ[defgnoxyz{????<
5?2
(?%
Decoder_Input??????????
p 

 
? ""?????????????
(__inference_decoder_layer_call_fn_403712? $%23<=>?FGPQRSZ[defgnoxyz{????<
5?2
(?%
Decoder_Input??????????
p

 
? ""?????????????
(__inference_decoder_layer_call_fn_403996? $%23<=>?FGPQRSZ[defgnoxyz{??8?5
.?+
!?
inputs??????????
p 

 
? ""?????????????
(__inference_decoder_layer_call_fn_404061? $%23<=>?FGPQRSZ[defgnoxyz{??8?5
.?+
!?
inputs??????????
p

 
? ""?????????????
$__inference_signature_wrapper_403931? $%23<=>?FGPQRSZ[defgnoxyz{??H?E
? 
>?;
9
Decoder_Input(?%
Decoder_Input??????????"Q?N
L
Conv2DTranspose_136?3
Conv2DTranspose_13???????????