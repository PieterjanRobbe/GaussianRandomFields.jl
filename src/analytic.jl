## analytic.jl :special cases where eigenfunctions are known analytically

# (separable) exponential covariance function with p = 0.5
function compute_analytic(cov::CovarianceFunction{1,Exponential{T}} where {T},n::N where {N<:Integer},pts::V where {V<:AbstractVector})
	λ = cov.cov.λ
    ω = findroots(λ, n)
    ev = @. 2*λ/(λ^2*ω^2+1)
    n = @. sqrt(2)/2*sqrt(1/ω*(λ^2*ω^2*cos(ω)*sin(ω)+λ^2*ω^3-2*λ*ω*cos.(ω)^2-cos(ω)*sin(ω)+ω)+2*λ)
	ef = diagm(1./n)*( sin.(ω*pts') + λ*diagm(ω)*cos.(ω*pts') )

	SpectralData(sqrt.(ev),ef')
end

# find all positive (>0) zeros of the transcendental function tan(ω) = 2*λ*ω/(λ^2*ω^2-1)
function findroots{T<:AbstractFloat,N<:Integer}(λ::T, n::N)

	# define the transcendental function
	f(ω) = (λ^2*ω^2-1)*sin(ω)-2*λ*ω*cos(ω)

	# find range around singularity 1/λ
	left_point_of_range = (2*floor(1/(π*λ)-1/2))*π/2 # left odd multiple of π/2
	right_point_of_range = (2*ceil(1/(π*λ)-1/2)+1)*π/2 # right odd multiple of π/2

	# find roots before 1/λ, if any
	if left_point_of_range ≠ π/2
		roots = zeros(min(n,round(UInt64,floor(abs(1/λ/π-1/2)))))
		left_point = π/2
		right_point = 1*π
		for i = 1:length(roots)
			@inbounds roots[i] = bisect_root(f,left_point,right_point)[1]
			right_point = roots[i] + π
			left_point = left_point + π
		end
	else
		roots = zeros(0)
	end

	# find roots inside range around 1/λ
	( length(roots) ≥ n || floor(1/(π*λ)-1/2) < 0 ) || 
	push!(roots,bisect_root(f,left_point_of_range+eps(T),1/λ)[1]) # first intersection point
	( length(roots) ≥ n || ceil(1/(π*λ)-1/2) < 0 ) || 
	push!(roots,bisect_root(f,1/λ,right_point_of_range)[1]) # second intersection point

	# if the first root is zero, cut it off
	roots[1] == 0 ? shift!(roots) : [] # empty expression

	# find roots after 1/λ
	startindex = 1 + length(roots)
	if n-length(roots) > 0
		roots = [roots; zeros(n-length(roots))]
		left_point = (2*ceil(1/(π*λ)-1/2)+2)*π/2
		right_point = (2*ceil(1/(π*λ)-1/2)+3)*π/2
		for i = startindex:length(roots)
			@inbounds roots[i] = bisect_root(f,left_point,right_point)[1]
			right_point = roots[i] + π
			left_point = left_point + π
		end
	end
	return roots
end

# bissection method to find the zeros of a function in a particular interval [x1,x2]
function bisect_root(fn::Function, x1::T, x2::T) where {T<:Real}
	xm = middle(x1, x2)
	s1 = sign(fn(x1))
	s2 = sign(fn(x2))
	while x1 < xm < x2
		sm =  sign(fn(xm))

		if s1 != sm
			x2 = xm
			s2 = sm
		else
			x1 = xm
			s1 = sm
		end

		xm = middle(x1, x2)
	end

	return x1, x2
end

# helper function to find the real "mid point" of two given floating point numbers
function middle(x1::Float64, x2::Float64)
	# use the usual float rules for combining non-finite numbers
	if !isfinite(x1) || !isfinite(x2)
		return x1 + x2
	end

	# always return 0.0 when inputs have opposite sign
	if sign(x1) != sign(x2) && x1 != 0.0 && x2 != 0.0
		return 0.0
	end

	negate = x1 < 0.0 || x2 < 0.0

	x1_int = reinterpret(UInt64, abs(x1))
	x2_int = reinterpret(UInt64, abs(x2))
	unsigned = reinterpret(Float64, (x1_int + x2_int) >> 1)

	negate ? -unsigned : unsigned
end
