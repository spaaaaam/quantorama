extern crate num_traits;
use num_traits::{Float, Num, NumCast, One, ToPrimitive, Zero};
use std::ops;
use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim, DimName, OVector, Scalar, storage::Owned};

// scalar jet
pub type SJet<T> = VJet<T, 1>;
// vector jet
pub type VJet<T, const N: usize> = Jet<T, Const<N>>;

#[derive(Copy, Clone, Debug)]
pub struct Jet<T:Float + Scalar, N:Dim + DimName>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    f: T,
    dfdx : OVector<T,N>
}

impl<T:Float + Scalar, N:Dim + DimName> Jet<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    pub fn new(f: T, dfdx: OVector<T, N>) -> Jet<T, N>
    {
        Jet{f, dfdx}
    }

    pub fn size(self) -> usize
    {
        self.dfdx.shape().0
    }

    pub fn from_slice(f: T, dfdx: &[T]) -> Jet<T, N>
    {
        Jet{f, dfdx:OVector::<T,N>::from_row_slice(dfdx)}
    }

    pub fn from_scalar(f: T, dfdx: T) -> Jet<T,N>
    {
        Jet::from_slice(f, &[dfdx])
    }

    #[allow(dead_code)]
    pub fn variable(value: T) -> Jet<T, N>
    {
        Jet::new(value, OVector::<T,N>::repeat(T::one()))
    }

    #[allow(dead_code)]
    pub fn variable_i(value: T, i: usize) -> Jet<T, N>
    {
        let mut jet = Jet::constant(value);
        assert!(i < jet.size());
        jet.dfdx[i] = T::one();
        jet
    }

    #[allow(dead_code)]
    pub fn constant(value: T) -> Jet<T, N>
    {
        Jet::new(value, OVector::<T,N>::zeros())
    }

    pub fn zero() -> Jet<T, N>
    {
        Jet::constant(T::zero())
    }

    pub fn one() -> Jet<T, N>
    {
        Jet::constant(T::one())
    }

    #[inline]
    pub fn map_jet<F>(&self, value: T, f: F) -> Jet<T, N>
    where
        F: Fn(&T) -> T
    {
        let dfdx = self.dfdx.map(|x| f(&x));
        Jet::new(value, dfdx)
    }

    #[inline]
    pub fn zip_map_jet<F>(&self, value: T, f: F, rhs: &Jet<T,N>) -> Jet<T, N>
    where
        F: Fn(&T, &T) -> T
    {
        let dfdx = self.dfdx.zip_map(&rhs.dfdx, |x, y| f(&x, &y));
        Jet::new(value, dfdx)
    }
}

impl<M: Copy + Default, T: Float + Scalar + Copy + float_cmp::ApproxEq<Margin=M>, N:Dim + DimName> float_cmp::ApproxEq for Jet<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    type Margin = M;

    fn approx_eq<U: Into<Self::Margin>>(self, other: Self, margin: U) -> bool 
    {
        let margin = margin.into();
        self.f.approx_eq(other.f, margin) 
            && self.dfdx.iter().zip(other.dfdx.iter())
            .all(|(&x,&y)|x.approx_eq(y, margin))
    }
}

impl<T:Float + Scalar, N:Dim + DimName> ops::Rem for Jet<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    type Output = Self;

    /// remainder is not a valid operation on dual numbers,
    /// but its implementation is required for the `Float` trait
    fn rem(self, _: Self) -> Self {
        unimplemented!()
    }
}

impl<T:Float + Scalar + From<f64> + Num, N:Dim + DimName> Num for Jet<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    type FromStrRadixErr = <T as Num>::FromStrRadixErr;
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        <T as Num>::from_str_radix(s, radix).map(From::from)
    }
}

impl<T: Float + Scalar + From<f64>, N:Dim + DimName> PartialOrd for Jet<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.f.partial_cmp(&other.f)
    }
}

impl<T: Float + Scalar + From<f64>, N:Dim + DimName> NumCast for Jet<T, N> 
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    fn from<P:ToPrimitive>(n: P) -> Option<Self>
    {
        // try to cast P to an f64, then used
        // to initialize a constant Jet value
        <T as NumCast>::from(n).map(<Self as From<T>>::from)
    }
}

impl<T:Float + Scalar + From<f64>, N:Dim + DimName> ToPrimitive for Jet<T, N> 
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    fn to_i64(&self) -> Option<i64> {
        self.f.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.f.to_u64()
    }
}

impl<T:Float + Scalar + From<f64> + One, N:Dim + DimName> One for Jet<T, N> 
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    fn one() -> Self {
        From::from(<T as One>::one())
    }
}

impl<T:Float + Scalar + From<f64> + Zero, N:Dim + DimName> Zero for Jet<T, N> 
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    fn zero() -> Self {
        From::from(<T as Zero>::zero())
    }

    fn is_zero(&self) -> bool {
        self.f.is_zero() && self.dfdx.iter().all(|x| x.is_zero())
    }
}

impl<T:Float + Scalar + From<f64>, N:Dim + DimName> Float for Jet<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    fn nan() -> Self {
        From::from(T::nan())
    }

    fn infinity() -> Self {
        From::from(T::infinity())
    }

    fn neg_infinity() -> Self {
        From::from(T::neg_infinity())
    }

    fn neg_zero() -> Self {
        From::from(T::neg_zero())
    }

    fn min_value() -> Self {
        From::from(T::min_value())
    }

    fn min_positive_value() -> Self {
        From::from(T::min_positive_value())
    }

    fn epsilon() -> Self {
        From::from(T::epsilon())
    }

    fn max_value() -> Self {
        From::from(T::max_value())
    }

    fn is_nan(self) -> bool {
        self.f.is_nan() || self.dfdx.iter().all(|x| x.is_nan())
    }

    fn is_infinite(self) -> bool {
        self.f.is_infinite() || self.dfdx.iter().all(|x| x.is_infinite())
    }

    fn is_finite(self) -> bool {
        self.f.is_finite() && self.dfdx.iter().all(|x| x.is_finite())
    }

    fn is_normal(self) -> bool {
        self.f.is_normal() && self.dfdx.iter().all(|x| x.is_normal())
    }

    fn classify(self) -> std::num::FpCategory {
        self.f.classify()
    }

    fn floor(self) -> Self {
        Jet::new(self.f.floor(), self.dfdx)
    }

    fn ceil(self) -> Self {
        Jet::new(self.f.ceil(), self.dfdx)
    }

    fn round(self) -> Self {
        Jet::new(self.f.round(), self.dfdx)
    }

    fn trunc(self) -> Self {
        Jet::new(self.f.trunc(), self.dfdx)
    }

    fn fract(self) -> Self {
        Jet::new(self.f.fract(), self.dfdx)
    }

    fn abs(self) -> Self {
        Jet::new(self.f.abs(), 
            if self.f>=From::from(0.) {
                    self.dfdx
                } else {
                    -self.dfdx
                })
    }

    fn signum(self) -> Self {
        From::from(self.f.signum())
    }

    fn is_sign_positive(self) -> bool {
        self.f.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.f.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        let mut jet = Jet::constant(self.f.mul_add(a.f, b.f));
        for i in 0..self.size()
        {
            jet.dfdx[i] = a.f * self.dfdx[i] + a.dfdx[i] * self.f + b.dfdx[i]
        }
        jet
    }

    fn recip(self) -> Self {
        Jet::one()/self
    }

    fn powi(self, n: i32) -> Self {
        let pow_n_minus_one = self.f.powi(n-1);
        let n_pow_n_minus_one = <T as NumCast>::from(n).expect("Invalid value for integer power") * pow_n_minus_one;
        self.map_jet(self.f * pow_n_minus_one, |dfdx| *dfdx * n_pow_n_minus_one)
    }

    fn powf(self, n: Self) -> Self {
        let powf = self.f.powf(n.f);
        let a = n.f * self.f.powf(n.f-T::one());
        let b = powf * self.f.ln();
        self.zip_map_jet(powf, |dfdx, dndx| *dfdx * a + *dndx * b, &n)
    }

    fn sqrt(self) -> Self {
        let sqrt = self.f.sqrt();
        let a = T::one() / (<T as NumCast>::from(2u8).unwrap() * sqrt);
        self.map_jet(sqrt, |dfdx| *dfdx * a)
    }

    fn exp(self) -> Self {
        let exp = self.f.exp();
        self.map_jet(exp, |dfdx| *dfdx * exp)
    }

    fn exp2(self) -> Self {
        let exp2 = self.f.exp();
        let ln2 =  <T as NumCast>::from(2u8).unwrap().ln();
        self.map_jet(exp2, |dfdx| *dfdx * ln2 * exp2) // d/dx(u^f(x)) = ln(u) * dfdx * u^f(x)
    }

    fn ln(self) -> Self {
        let f = self.f;
        self.map_jet(f.ln(),|dfdx| *dfdx / f)
    }

    fn log(self, base: Self) -> Self {
        self.ln()/base.ln()
    }

    fn log2(self) -> Self {
        let ln2 = <T as NumCast>::from(2u8).unwrap().ln();
        self.map_jet(self.f.log2(), |dfdx| *dfdx/(ln2 * self.f))
    }

    fn log10(self) -> Self {
        let ln10 = <T as NumCast>::from(10u8).unwrap().ln();
        self.map_jet(self.f.log10(), |dfdx| *dfdx/(ln10 * self.f))
    }

    fn to_degrees(self) -> Self {
        let halfpi = Float::acos(Self::zero());
        let ninety = <Jet<T, N> as NumCast>::from(90u8).unwrap();
        self * ninety / halfpi
    }

    fn to_radians(self) -> Self {
        let halfpi = Float::acos(Self::zero());
        let ninety = <Jet<T, N> as NumCast>::from(90u8).unwrap();
        self * halfpi / ninety
    }

    fn max(self, other: Self) -> Self {
        if self.f < other.f {
            other
        } else {
            self
        }
    }

    fn min(self, other: Self) -> Self {
        if self.f > other.f {
            other
        } else {
            self
        }
    }

    fn abs_sub(self, other: Self) -> Self {
        let d = self - other;
        if d.f < <T as NumCast>::from(0u8).unwrap() {
            -d
        } else {
            d
        }
    }

    fn cbrt(self) -> Self {
        let cubic_root = self.f.cbrt();
        self.map_jet(cubic_root, |dfdx| *dfdx / (<T as NumCast>::from(3u8).unwrap() * cubic_root))
    }

    fn hypot(self, other: Self) -> Self {
        // d/dx(sqrt(x^2+y^2)) = (x*Dx + y*Dy) / sqrt(x^2 + y^2)
        let hypot = self.f.hypot(other.f);
        self.zip_map_jet(hypot, |dfdx, dgdx| (*dfdx * self.f + *dgdx *other.f)/hypot, &other)
    }

    fn sin(self) -> Self {
        let cos = self.f.cos();
        self.map_jet(self.f.sin(), |dfdx| *dfdx * cos)
    }

    fn cos(self) -> Self {
        let sin = self.f.sin();
        self.map_jet(self.f.cos(), |dfdx| dfdx.neg() * sin)
    }

    fn tan(self) -> Self {
        let tan = self.f.tan();
        let one_on_cos_sqr = T::one() + tan * tan;
        self.map_jet(tan, |dfdx| *dfdx * one_on_cos_sqr)
    }

    fn asin(self) -> Self {
        let a = (T::one()-self.f.powi(2)).sqrt();
        self.map_jet(self.f.asin(), |dfdx| *dfdx / a)
    }

    fn acos(self) -> Self {
        let a = (T::one()-self.f.powi(2)).sqrt();
        self.map_jet(self.f.acos(), |dfdx| dfdx.neg() / a)
    }

    fn atan(self) -> Self {
        let a = T::one()+self.f.powi(2);
        self.map_jet(self.f.atan(), |dfdx| *dfdx / a)
    }

    fn atan2(self, other: Self) -> Self {
        // d(atan(y/x)) = (xdy - ydx) / (x^2 + y^2)
        let norm_sqr = self.f.powi(2)+other.f.powi(2);
        self.zip_map_jet(self.f.atan2(other.f), |dfdx, dgdx| (*dfdx * other.f - *dgdx * self.f)/norm_sqr, &other)
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.f.sin_cos();
        let sin_jet = self.map_jet(sin, |dfdx| *dfdx * cos);
        let cos_jet = self.map_jet(cos, |dfdx| dfdx.neg() * sin);
        (sin_jet, cos_jet)
    }

    fn exp_m1(self) -> Self {
        let exp = self.f.exp();
        self.map_jet(self.f.exp_m1(), |dfdx| *dfdx * exp)
    }

    fn ln_1p(self) -> Self {
        let one_over_one_plus_f = T::one() / (T::one() + self.f);
        self.map_jet(self.f.ln_1p(), |dfdx| *dfdx * one_over_one_plus_f)
    }

    fn sinh(self) -> Self {
        let cosh = self.f.cosh();
        self.map_jet(self.f.sinh(), |dfdx| *dfdx * cosh)
    }

    fn cosh(self) -> Self {
        let sinh = self.f.sinh();
        self.map_jet(self.f.cosh(), |dfdx| *dfdx * sinh)
    }

    fn tanh(self) -> Self {
        let tanh = self.f.tanh();
        let one_minus_tanh_sqr = T::one() - tanh*tanh;
        self.map_jet(tanh, |dfdx| *dfdx * one_minus_tanh_sqr)
    }

    fn asinh(self) -> Self {
        let a = T::one()/(T::one() + self.f.powi(2)).sqrt();
        self.map_jet(self.f.asinh(), |dfdx| *dfdx * a)
    }

    fn acosh(self) -> Self {
        let a = T::one() / ((self.f + T::one()).sqrt() * (self.f - T::one()).sqrt());
        self.map_jet(self.f.acosh(), |dfdx| *dfdx * a)
    }

    fn atanh(self) -> Self {
        let a = T::one() / (T::one() - self.f.powi(2));
        self.map_jet(self.f.atanh(), |dfdx| *dfdx * a)
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.f.integer_decode()
    }
}

impl<T:Float + Scalar, N:Dim + DimName> ops::Neg for Jet<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    type Output = Self;
    fn neg(self) -> Self{
        Jet::new(self.f.neg(),self.dfdx.neg())
    }
}

impl<T:Float + Scalar + From<f64>, N:Dim + DimName> From<T> for Jet<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    fn from(x: T) -> Jet<T, N> 
    {
        Jet::constant(x)
    }
}

impl<T:Float + Scalar + From<f64>, N:Dim + DimName> PartialEq for Jet<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    fn eq(&self, other: &Jet<T, N>) -> bool 
    { 
        self.f == other.f && self.dfdx == other.dfdx
    }
}

impl<T:Float + Scalar, N:Dim + DimName> ops::Add<Jet<T, N>> for Jet<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    type Output = Jet<T, N>;

    fn add(self, _rhs: Jet<T, N>) -> Jet<T, N>
    {
        self.zip_map_jet(self.f + _rhs.f, |dfdx, dgdx| *dfdx + *dgdx, &_rhs)
    }
}

impl<T:Float + Scalar, N:Dim + DimName> ops::Sub<Jet<T, N>> for Jet<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    type Output = Jet<T, N>;

    fn sub(self, _rhs: Jet<T, N>) -> Jet<T, N>
    {
        self.zip_map_jet(self.f - _rhs.f, |dfdx, dgdx| *dfdx - *dgdx, &_rhs)
    }
}

impl<T:Float + Scalar, N:Dim + DimName> ops::Mul<Jet<T, N>> for Jet<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    type Output = Jet<T, N>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, _rhs: Jet<T, N>) -> Jet<T, N>
    {
        self.zip_map_jet(self.f * _rhs.f, |dfdx, dgdx| *dfdx * _rhs.f + *dgdx * self.f, &_rhs)
    }
}

impl<T:Float + Scalar, N:Dim + DimName> ops::Div<Jet<T, N>> for Jet<T, N>
where
    DefaultAllocator: Allocator<T, N>,
    Owned<T, N>: Copy
{
    type Output = Jet<T, N>;

    fn div(self, _rhs: Jet<T, N>) -> Jet<T, N>
    {
        let denom = _rhs.f * _rhs.f;
        self.zip_map_jet(self.f / _rhs.f, |dfdx, dgdx| (*dfdx * _rhs.f - *dgdx * self.f)/denom, &_rhs)
    }
}

#[cfg(test)]
use float_cmp::ApproxEq;

#[cfg(test)]
pub(crate) const SMALL_TOLERANCE : float_cmp::F64Margin = float_cmp::F64Margin{ ulps: 1, epsilon: 0.0 };
#[cfg(test)]
pub(crate) const MEDIUM_TOLERANCE : float_cmp::F64Margin = float_cmp::F64Margin{ ulps: 3, epsilon: 0.0 };
// const LARGE_TOLERANCE : float_cmp::F64Margin = float_cmp::F64Margin{ ulps: 10, epsilon: 0.0 };

#[cfg(test)]
macro_rules! assert_close {
    // TODO: display values when not close enough
    ($a:expr, $b:expr) => {
        assert!($a.approx_eq($b, SMALL_TOLERANCE))
    };
    ($a:expr, $b:expr, $tol:expr) => {
        assert!($a.approx_eq($b, $tol))
    };
}

#[test]
fn test_constant() {
    let manual_cst = Jet::new(1., OVector::<f64,Const<1>>::from_row_slice(&[0.]));
    println!("{:?}", manual_cst);

    let cst = Jet::constant(1.);
    println!("{:?}", cst);

    assert_eq!(manual_cst, cst);
}

#[test]
fn test_identity()
{
    let naive_var = Jet::new(0., OVector::<f64,Const<1>>::from_row_slice(&[1.]));
    println!("{:?}", naive_var);

    let var = SJet::variable(0.0);
    println!("{:?}", var);

    assert_eq!(naive_var, var);
}

#[test]
fn test_log()
{
    let log_jet = SJet::variable(1.0).ln();
    println!("{:?}", log_jet);

    let reference = Jet::from_scalar(0., 1.);

    assert_eq!(log_jet, reference);
}

#[test]
fn test_exp()
{
    let exp_jet = SJet::variable(1.0).exp();
    println!("{:?}", exp_jet);

    let reference = Jet::from_scalar(std::f64::consts::E, std::f64::consts::E);

    assert_eq!(exp_jet, reference);
}

#[test]
fn test_powi()
{
    let pow_jet = SJet::variable(3.0).powi(2);
    println!("{:?}", pow_jet);

    let reference = Jet::from_scalar(9.0, 6.0);

    assert_eq!(pow_jet, reference);
}

#[test]
fn test_powf()
{
    let pow_jet = SJet::variable(3.0).powf(Jet::constant(0.5));
    println!("{:?}", pow_jet);

    let reference = Jet::from_scalar(f64::sqrt(3.0), 0.5/f64::sqrt(3.0));

    assert_close!(pow_jet, reference);
    
}

#[test]
fn test_powi_vs_multiply()
{
    let x0 = 3.0;
    let pow_jet = SJet::variable(x0).powi(2);
    println!("{:?}", pow_jet);

    let mul_jet = SJet::variable(x0)*SJet::variable(x0);

    assert_eq!(pow_jet, mul_jet);
}

#[test]
fn test_powf_vs_sqrt()
{
    let x0 = 4.0;
    let pow_jet = SJet::variable(x0).powf(Jet::constant(0.5));
    println!("{:?}", pow_jet);

    let sqrt_jet = SJet::variable(x0).sqrt();

    assert_eq!(pow_jet, sqrt_jet);
}

#[test]
fn test_identity_sqrt_pow2()
{
    let x0 = 4.0;

    assert_eq!(SJet::variable(x0).powi(2).sqrt(), SJet::variable(x0));
    assert_eq!(SJet::variable(x0).sqrt().powi(2), SJet::variable(x0));
}

#[test]
fn test_identity_log_exp()
{
    let x0 = 4.0;

    assert_eq!(SJet::variable(x0).ln().exp(), SJet::variable(x0));
    assert_eq!(SJet::variable(x0).exp().ln(), SJet::variable(x0));
}

#[test]
fn test_identity_cos_acos()
{
    let x0 = 0.3;

    assert_close!(SJet::variable(x0).cos().acos(), SJet::variable(x0), MEDIUM_TOLERANCE);
    assert_close!(SJet::variable(x0).acos().cos(), SJet::variable(x0), MEDIUM_TOLERANCE);
}

#[test]
fn test_identity_sin_asin()
{
    let x0 = 0.3;

    assert_close!(SJet::variable(x0).sin().asin(), SJet::variable(x0));
    assert_close!(SJet::variable(x0).asin().sin(), SJet::variable(x0));
}

#[test]
fn test_identity_tan_atan()
{
    let x0 = 0.3;

    assert_close!(SJet::variable(x0).tan().atan(), SJet::variable(x0));
    assert_close!(SJet::variable(x0).atan().tan(), SJet::variable(x0));
}

#[test]
fn test_identity_cosh_acosh()
{
    let x0 = 1.3;

    assert_close!(SJet::variable(x0).cosh().acosh(), SJet::variable(x0));
    assert_close!(SJet::variable(x0).acosh().cosh(), SJet::variable(x0));
}

#[test]
fn test_identity_sinh_asinh()
{
    let x0 = 0.3;

    assert_close!(SJet::variable(x0).sinh().asinh(), SJet::variable(x0));
    assert_close!(SJet::variable(x0).asinh().sinh(), SJet::variable(x0));
}

#[test]
fn test_identity_tanh_atanh()
{
    let x0 = 0.3;

    assert_close!(SJet::variable(x0).tanh().atanh(), SJet::variable(x0), MEDIUM_TOLERANCE);
    assert_close!(SJet::variable(x0).atanh().tanh(), SJet::variable(x0), MEDIUM_TOLERANCE);
}

#[test]
fn test_multivariate_function()
{
    let x = VJet::<f64, 2>::from_slice(3.0, &[1.0, 0.0]);
    let y = VJet::<f64, 2>::from_slice(2.0, &[0.0, 1.0]);

    let f = (x+y).powi(2);

    assert_eq!(f.f, 25.0);
    assert_eq!(f.dfdx.shape(), (2,1));
    assert_eq!(f.dfdx[0], 10.0);
}

#[test]
#[should_panic]
fn test_multivariate_throws()
{
    // test to ensure bad scalar jet instanciation throws
    let _ = SJet::from_slice(2.0, &[0.0, 1.0]);
}

#[test]
fn test_monovariate_function()
{
    let x = SJet::variable(3.0);

    let f = (x+x).powi(2);

    assert_eq!(f.f, 36.0);
    assert_eq!(f.dfdx[0], 24.0);
}

#[test]
fn test_other_monovariate_function()
{
    let x = SJet::variable(3.0);

    let f = (Jet::constant(2.)*x).powi(2);

    assert_eq!(f.f, 36.0);
    assert_eq!(f.dfdx[0], 24.0);
}

#[test]
fn test_simpler_monovariate_function()
{
    let x = SJet::variable(6.0);

    let f = (x).powi(2);

    assert_eq!(f.f, 36.0);
    assert_eq!(f.dfdx[0], 12.0);
}

#[test]
fn test_vector_function()
{
    let x = SJet::variable(2.0);
    let f = vec![x.sqrt(), x.ln()];

    assert_eq!(f[0].f, f64::sqrt(2.0));
    assert_eq!(f[0].dfdx[0], 0.5/f64::sqrt(2.0));
    assert_eq!(f[1].f, f64::ln(2.0));
    assert_eq!(f[1].dfdx[0], 0.5);   
}

// #[test]
// fn test_multivariate_vector_function()
// {

// }

// use std::vec::Vec;
// fn cost_function<T>(parameters: Vec<T>, residuals: &Vec<T>)-> bool{
//     return false;
// }


// extern crate nalgebra as na;
// use na::DMatrix;
// pub struct CostFunction<T:Float>{
//     parameter_count: usize,
//     residual_count: usize
// }
// pub trait Jacobian<T:Float>{
//     fn compute_dfidxj(r:usize, c:usize) -> T;
//     fn compute_jacobian(& self) -> na::DMatrix<T>;
// }

// impl Jacobian<T> for CostFunction<T:Float>{
//     fn compute_dfidxj(r:usize, c:usize) -> T {
        
//     }

//     fn compute_jacobian(& self) -> na::DMatrix<T> {
//         return DMatrix::from_fn(self.residual_count, self.parameter_count, compute_dfidxj);
//     }
// }