extern crate num_traits;
use num_traits::{Float, Num, NumCast, One, ToPrimitive, Zero};
use std::ops;

#[derive(Copy,Clone, Debug)]
pub struct Jet<T:Float>
{
    f: T,
    dfdx : T
}

impl<T:Float> Jet<T>
{
    pub fn new(f: T, dfdx: T) -> Jet<T>
    {
        Jet{f, dfdx}
    }

    #[allow(dead_code)]
    fn variable(value: T) -> Jet<T>
    {
        Jet::new(value, T::one())
    }

    #[allow(dead_code)]
    fn constant(value: T) -> Jet<T>
    {
        Jet::new(value, T::zero())
    }
}

impl<M: Copy + Default, T: Float + Copy + float_cmp::ApproxEq<Margin=M>> float_cmp::ApproxEq for Jet<T>
{
    type Margin = M;

    fn approx_eq<N: Into<Self::Margin>>(self, other: Self, margin: N) -> bool 
    {
        let margin = margin.into();
        self.f.approx_eq(other.f, margin) && self.dfdx.approx_eq(other.dfdx, margin)
    }
}

// https://docs.rs/num-traits/0.2.0/num_traits/float/trait.Float.html

impl<T:Float> ops::Rem for Jet<T>
{
    type Output = Self;

    /// **UNIMPLEMENTED!!!**
    ///
    /// As far as I know, remainder is not a valid operation on dual numbers,
    /// but is required for the `Float` trait to be implemented.
    fn rem(self, _: Self) -> Self {
        unimplemented!()
    }
}

impl<T:Float+From<f64>+Num> Num for Jet<T>
{
    type FromStrRadixErr = <T as Num>::FromStrRadixErr;
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        <T as Num>::from_str_radix(s, radix).map(From::from)
    }
}

impl<T:Float+From<f64>> PartialOrd for Jet<T>
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.f.partial_cmp(&other.f)
    }
}

impl<T:Float+From<f64>> NumCast for Jet<T> {
    fn from<P:ToPrimitive>(n: P) -> Option<Self>
    {
        // We first try to cast P to an f64, and then use this
        // to initialize a constant AutoDiff value.
        <T as NumCast>::from(n).map(<Self as From<T>>::from)
    }
}

impl<T:Float+From<f64>> ToPrimitive for Jet<T> {
    fn to_i64(&self) -> Option<i64> {
        self.f.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.f.to_u64()
    }
}

impl<T:Float + From<f64> + One> One for Jet<T> {
    fn one() -> Self {
        From::from(<T as One>::one())
    }
}

impl<T:Float + From<f64> + Zero> Zero for Jet<T> {
    fn zero() -> Self {
        From::from(<T as Zero>::zero())
    }

    fn is_zero(&self) -> bool {
        self.f.is_zero() && self.dfdx.is_zero()
    }
}

impl<T:Float+From<f64>> Float for Jet<T>
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
        self.f.is_nan() || self.dfdx.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.f.is_infinite() || self.dfdx.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.f.is_finite() && self.dfdx.is_finite()
    }

    fn is_normal(self) -> bool {
        self.f.is_normal() && self.dfdx.is_normal()
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
        Jet::new(self.f.mul_add(a.f, b.f),
        a.f * self.dfdx + a.dfdx * self.f + b.dfdx)
    }

    fn recip(self) -> Self {
        let f = self.f;
        Jet::new(f.recip(),-self.dfdx/(f*f))
    }

    fn powi(self, n: i32) -> Self {
        Jet::new(self.f.powi(n),
        <T as NumCast>::from(n).unwrap()*self.dfdx*self.f.powi(n-1))
    }

    fn powf(self, n: Self) -> Self {
        // TODO: no diff wrt. n
        Jet::new(self.f.powf(n.f),
        <T as NumCast>::from(n.f).unwrap()*self.dfdx*self.f.powf(n.f-T::one()))
    }

    fn sqrt(self) -> Self {
        Jet::new(self.f.sqrt(),self.dfdx/(<T as From<f64>>::from(2.)*self.f.sqrt()))
    }

    fn exp(self) -> Self {
        Jet::new(self.f.exp(),
        self.dfdx*self.f.exp())
    }

    fn exp2(self) -> Self {
        let b = self.f.exp2();
        Jet::new(b, <T as NumCast>::from(2u8).unwrap().ln() * self.dfdx * b) // d/dx(u^f(x)) = ln(u) * dfdx * u^f(x)
    }

    fn ln(self) -> Self {
        Jet::new(self.f.ln(),self.dfdx / self.f)
    }

    fn log(self, base: Self) -> Self {
        Jet::new(self.f.log(base.f),self.dfdx / (base.f.ln() * self.f))
    }

    fn log2(self) -> Self {
        Jet::new(self.f.log2(),self.dfdx/(<T as From<f64>>::from(2.).ln() * self.f))
    }

    fn log10(self) -> Self {
        Jet::new(self.f.log10(),self.dfdx/(<T as From<f64>>::from(10.).ln() * self.f))
    }

    fn to_degrees(self) -> Self {
        let halfpi = Float::acos(Self::zero());
        let ninety = <Jet<T> as NumCast>::from(90u8).unwrap();
        self * ninety / halfpi
    }

    fn to_radians(self) -> Self {
        let halfpi = Float::acos(Self::zero());
        let ninety = <Jet<T> as NumCast>::from(90u8).unwrap();
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
        Jet::new(self.f.cbrt(), self.dfdx * self.f.powf(From::from(-2./3.)) / <T as NumCast>::from(3u8).unwrap())
    }

    fn hypot(self, other: Self) -> Self {
        // d/dx(sqrt(x^2+y^2)) = (x*Dx + y*Dy) / sqrt(x^2 + y^2)
        Jet::new(self.f.hypot(other.f),
        (self.f*self.dfdx + other.f*other.dfdx)/(self.f.powi(2)+other.f.powi(2)).sqrt())
    }

    fn sin(self) -> Self {
        Jet::new(self.f.sin(),self.dfdx * self.f.cos())
    }

    fn cos(self) -> Self {
        Jet::new(self.f.cos(),- self.dfdx * self.f.sin())
    }

    fn tan(self) -> Self {
        Jet::new(self.f.tan(),self.dfdx/self.f.cos().powi(2))
    }

    fn asin(self) -> Self {
        Jet::new(self.f.asin(), self.dfdx/(T::one()-self.f.powi(2)).sqrt())
    }

    fn acos(self) -> Self {
        Jet::new(self.f.acos(),self.dfdx.neg()/(T::one()-self.f.powi(2)).sqrt())
    }

    fn atan(self) -> Self {
        Jet::new(self.f.atan(), self.dfdx/(T::one()+self.f.powi(2)))
    }

    fn atan2(self, other: Self) -> Self {
        // d(atan(y/x)) = (xdy - ydx) / (x^2 + y^2)
        Jet::new(self.f.atan2(other.f),(self.f*other.dfdx - other.f*self.dfdx)/(self.f.powi(2)+other.f.powi(2)))
    }

    fn sin_cos(self) -> (Self, Self) {
        (<Jet<_> as Float>::sin(self), <Jet<_> as Float>::cos(self))
    }

    fn exp_m1(self) -> Self {
        Jet::new(self.f.exp_m1(),self.dfdx * self.f.exp())
    }

    fn ln_1p(self) -> Self {
        Jet::new(self.f.ln_1p(),self.dfdx / (T::one() + self.f))
    }

    fn sinh(self) -> Self {
        Jet::new(self.f.sinh(),self.dfdx * self.f.cosh())
    }

    fn cosh(self) -> Self {
        Jet::new(self.f.cosh(),self.dfdx * self.f.sinh())
    }

    fn tanh(self) -> Self {
        Jet::new(self.f.tanh(),self.dfdx * (T::one() - self.f.tanh().powi(2)))
    }

    fn asinh(self) -> Self {
        Jet::new(self.f.asinh(), self.dfdx / (T::one() + self.f.powi(2)).sqrt())
    }

    fn acosh(self) -> Self {
        Jet::new(self.f.acosh(), self.dfdx / (self.f.powi(2) - <T as From<f64>>::from(1.0)).sqrt())
    }

    fn atanh(self) -> Self {
        Jet::new(self.f.atanh(), self.dfdx / (<T as From<f64>>::from(1.0) - self.f.powi(2)))
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.f.integer_decode()
    }
}

impl<T:Float> ops::Neg for Jet<T>{
    type Output = Self;
    fn neg(self) -> Self{
        Jet::new(self.f.neg(),self.dfdx.neg())
    }
}

impl<T:Float+From<f64>> From<T> for Jet<T>
{
    fn from(x: T) -> Jet<T> 
    {
        Jet{f: x, dfdx: <T as From<f64>>::from(0.)}
    }
}

impl<T:Float+From<f64>> PartialEq for Jet<T>
{
    fn eq(&self, other: &Jet<T>) -> bool 
    { 
        self.f == other.f && self.dfdx == other.dfdx
    }
}

impl<T:Float> ops::Add<Jet<T>> for Jet<T>
{
    type Output = Jet<T>;

    fn add(self, _rhs: Jet<T>) -> Jet<T>
    {
        Jet::new(self.f + _rhs.f, self.dfdx + _rhs.dfdx)
    }
}

impl<T:Float> ops::Sub<Jet<T>> for Jet<T>
{
    type Output = Jet<T>;

    fn sub(self, _rhs: Jet<T>) -> Jet<T>
    {
        Jet::new(self.f - _rhs.f, self.dfdx - _rhs.dfdx)
    }
}

impl<T:Float> ops::Mul<Jet<T>> for Jet<T>
{
    type Output = Jet<T>;

    fn mul(self, _rhs: Jet<T>) -> Jet<T>
    {
        Jet::new(self.f * _rhs.f, self.dfdx *_rhs.f + self.f * _rhs.dfdx)
    }
}

impl<T:Float> ops::Div<Jet<T>> for Jet<T>
{
    type Output = Jet<T>;

    fn div(self, _rhs: Jet<T>) -> Jet<T>
    {
        Jet::new(self.f / _rhs.f, (self.dfdx *_rhs.f - self.f * _rhs.dfdx)/(_rhs.f * _rhs.f))
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
    let naive_cst = Jet::new(1., 0.);
    println!("{:?}", naive_cst);

    let cst = Jet::constant(1.);
    println!("{:?}", cst);

    assert_eq!(naive_cst, cst);
}

#[test]
fn test_identity()
{
    let naive_var = Jet::new(0., 1.);
    println!("{:?}", naive_var);

    let var = Jet::variable(0.0);
    println!("{:?}", var);

    assert_eq!(naive_var, var);
}

#[test]
fn test_log()
{
    let log_jet = Float::ln(Jet::variable(1.0));
    println!("{:?}", log_jet);

    let reference = Jet::new(0., 1.);

    assert_eq!(log_jet, reference);
}

#[test]
fn test_exp()
{
    let exp_jet = Jet::variable(1.0).exp();
    println!("{:?}", exp_jet);

    let reference = Jet::new(std::f64::consts::E, std::f64::consts::E);

    assert_eq!(exp_jet, reference);
}

#[test]
fn test_powi()
{
    let pow_jet = Jet::variable(3.0).powi(2);
    println!("{:?}", pow_jet);

    let reference = Jet::new(9.0, 6.0);

    assert_eq!(pow_jet, reference);
}

#[test]
fn test_powf()
{
    let pow_jet = Jet::variable(3.0).powf(Jet::constant(0.5));
    println!("{:?}", pow_jet);

    let reference = Jet::new(f64::sqrt(3.0), 0.5/f64::sqrt(3.0));

    assert_close!(pow_jet, reference);
    
}

#[test]
fn test_powi_vs_multiply()
{
    let x0 = 3.0;
    let pow_jet = Jet::variable(x0).powi(2);
    println!("{:?}", pow_jet);

    let mul_jet = Jet::variable(x0)*Jet::variable(x0);

    assert_eq!(pow_jet, mul_jet);
}

#[test]
fn test_powf_vs_sqrt()
{
    let x0 = 4.0;
    let pow_jet = Jet::variable(x0).powf(Jet::constant(0.5));
    println!("{:?}", pow_jet);

    let sqrt_jet = Jet::variable(x0).sqrt();

    assert_eq!(pow_jet, sqrt_jet);
}

#[test]
fn test_identity_sqrt_pow2()
{
    let x0 = 4.0;

    assert_eq!(Jet::variable(x0).powi(2).sqrt(), Jet::variable(x0));
    assert_eq!(Jet::variable(x0).sqrt().powi(2), Jet::variable(x0));
}

#[test]
fn test_identity_log_exp()
{
    let x0 = 4.0;

    assert_eq!(Jet::variable(x0).ln().exp(), Jet::variable(x0));
    assert_eq!(Jet::variable(x0).exp().ln(), Jet::variable(x0));
}

#[test]
fn test_identity_cos_acos()
{
    let x0 = 0.3;

    assert_close!(Jet::variable(x0).cos().acos(), Jet::variable(x0), MEDIUM_TOLERANCE);
    assert_close!(Jet::variable(x0).acos().cos(), Jet::variable(x0), MEDIUM_TOLERANCE);
}

#[test]
fn test_identity_sin_asin()
{
    let x0 = 0.3;

    assert_close!(Jet::variable(x0).sin().asin(), Jet::variable(x0));
    assert_close!(Jet::variable(x0).asin().sin(), Jet::variable(x0));
}

#[test]
fn test_identity_tan_atan()
{
    let x0 = 0.3;

    assert_close!(Jet::variable(x0).tan().atan(), Jet::variable(x0));
    assert_close!(Jet::variable(x0).atan().tan(), Jet::variable(x0));
}

#[test]
fn test_identity_cosh_acosh()
{
    let x0 = 1.3;

    assert_close!(Jet::variable(x0).cosh().acosh(), Jet::variable(x0));
    assert_close!(Jet::variable(x0).acosh().cosh(), Jet::variable(x0));
}

#[test]
fn test_identity_sinh_asinh()
{
    let x0 = 0.3;

    assert_close!(Jet::variable(x0).sinh().asinh(), Jet::variable(x0));
    assert_close!(Jet::variable(x0).asinh().sinh(), Jet::variable(x0));
}

#[test]
fn test_identity_tanh_atanh()
{
    let x0 = 0.3;

    assert_close!(Jet::variable(x0).tanh().atanh(), Jet::variable(x0), MEDIUM_TOLERANCE);
    assert_close!(Jet::variable(x0).atanh().tanh(), Jet::variable(x0), MEDIUM_TOLERANCE);
}