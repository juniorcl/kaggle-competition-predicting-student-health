# Salve como teste_simples.jl
using Pkg
Pkg.activate(".")

using MLJ
using LightGBM
using Parquet2
using DataFrames


X_train = DataFrame(Parquet2.Dataset("../data/processed/X_train_fe.parquet"))

y_raw = DataFrame(Parquet2.Dataset("../data/interim/y_train.parquet"))
y_train = categorical(y_raw[:, 1])


X_train = coerce(X_train, 
    Count => Continuous,     # IDs ou inteiros que são números
    Textual => Multiclass,   # Strings -> Categóricas
    Integer => Continuous
)

y_train = categorical(y_train)

LGBMClassifier = @load LGBMClassifier pkg=LightGBM
model = LGBMClassifier()

mach = machine(model, X_train, y_train, scitype_check_level=0)
MLJ.fit!(mach)