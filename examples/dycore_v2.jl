# Provide function templates to Lucas and Jeremy
# ============================================== 

# Jeremy github branch for editing source, flux functions 

function fluxfun!(flux, Qstate, gradQstate, X ,Cstate)
end

function sourcefun!(source, Qstate, X, Cstate)
end

function bcfun!(Qstate, gradQstate, bcid, normals, X, Cstate)
end

function gradbcfun!(Qstate, bcid, normals, X, Cstate)
end

function cfun!(Qstate)
end
