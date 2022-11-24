import numpy as np
import numpy.random as npr
import numba as nb
tol = 1e-20
exp = np.exp
log = np.log
pi = np.pi
sin = np.sin
sqrt = np.sqrt

@nb.njit(nb.f8(nb.f8), fastmath=True)
def slog(x):
    return np.log(x + tol)

@nb.njit(nb.f8(nb.f8, nb.f8), fastmath=True)
def sdiv(x, y):
    return x/(y + tol)

@nb.njit(nb.f8(nb.f8, nb.f8, nb.f8, nb.f8), fastmath=True)
def gen_U(w1, w2, w3, gamma):
    V = npr.rand()
    W_p = npr.rand()
    if gamma >= 1:
        if (V < sdiv(w1, w1+w2)):
            U = abs(npr.randn()) / sqrt(gamma)
        else:
            U = pi * (1 - W_p**2)
    else:
        if (V < sdiv(w3,w3 + w2)):
            U = pi * W_p
        else:
            U = pi * (1 - W_p**2)
    return U

@nb.njit(nb.f8(nb.f8), fastmath=True)
def sinc(x):
    return sdiv(np.sin(x), x)

@nb.njit(nb.f8(nb.f8, nb.f8), fastmath=True)
def ratio_B(x, sigma):
    return sdiv(sdiv(sinc(x), sinc(sigma*x)**sigma),
            sinc((1-sigma)*x)**(1-sigma))

@nb.njit(nb.f8(nb.f8, nb.f8), fastmath=True)
def zolotarev(u, sigma):
    expn = min(sdiv(1, 1-sigma), 50.0)
    x = sdiv(sin(sigma*u)**sigma * sin((1-sigma)*u)**(1-sigma), sin(u))**expn
    return x

@nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8, nb.i4), fastmath=True)
def etstablernd(V0, alpha, tau, n):

    # check params
    if alpha <= 0 or alpha >= 1:
        raise ValueError('alpha must be in ]0,1[')
    if tau < 0:
        raise ValueError('tau must be >= 0')
    if V0 <= 0:
        raise ValueError('V0 must be > 0')

    lambda_alpha = tau**alpha * V0

    # Now we sample from an exponentially tilted distribution of parameters
    # sigma, lambda, as in (Devroye, 2009)
    gamma = lambda_alpha * alpha * (1-alpha)

    xi = 1/pi *((2+sqrt(pi/2)) * sqrt(2*gamma) + 1) # Correction in Hofert
    psi = 1/pi * exp(-gamma * pi**2/8) * (2 + sqrt(pi/2)) * sqrt(gamma * pi)
    w1 = xi * sqrt(pi/2/gamma)
    w2 = 2 * psi * sqrt(pi)
    w3 = xi * pi
    b = (1-alpha)/alpha

    samples = np.zeros(n)
    for i in range(n):
        while True:
            # generate U with density g*/G*
            while True:
                # Generate U with density proportional to g**
                U = gen_U(w1, w2, w3, gamma)
                while U >= pi:
                    U = gen_U(w1, w2, w3, gamma)

                assert U > 0
                assert U < pi

                W = npr.rand()
                zeta = sqrt(ratio_B(U, alpha))

                z = sdiv(1, 1 - (1 + sdiv(alpha*zeta, sqrt(gamma)))**sdiv(-1, alpha))
                rho = 1
                rho = pi * exp(min(-lambda_alpha * (1-zeta**(-2)), 1e+2)) \
                        * (xi * exp(-gamma*U**2/2) * (gamma>=1) \
                        + sdiv(psi, sqrt(pi-U)) \
                        + sdiv(xi * (gamma<1),
                            (1 + sqrt(pi/2))*sdiv(sqrt(gamma),zeta)+z))

                if W*rho <= 1:
                    break

            # Generate X with density proportional to g(x, U)
            a = zolotarev(U, alpha)
            m = sdiv(b,a)**alpha * lambda_alpha
            delta = sqrt(m*sdiv(alpha, a))
            a1 = delta * sqrt(pi/2)
            a2 = a1 + delta # correction in Hofert
            a3 = sdiv(z, a)
            s = a1 + delta + a3 # correction in Hofert
            V_p = npr.rand()
            N_p = npr.randn()
            E_p = -log(npr.rand())

            if V_p < sdiv(a1, s):
                X = m - delta*abs(N_p)
            elif V_p < sdiv(a2, s):
                X = delta * npr.rand() + m
            else:
                X = m + delta + a3 * E_p

            if X >= 0:
                E = -slog(npr.rand())
                cond = (a*(X-m) + exp(1/alpha*slog(lambda_alpha)-b*slog(m)) \
                        *(sdiv(m, X)**b - 1) - (N_p**2/2) * (X<m) - E_p * (X>m+delta))
                if cond <= E:
                    break

        samples[i] = exp(1/alpha*log(V0) - b*slog(X)) # more stable than V0^(1/alpha) * X**(-b)

    return samples

if __name__ == '__main__':

    x = etstablernd(1.0, 0.2, 0.0, 1)
