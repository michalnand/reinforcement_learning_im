import numpy

def kl_divergence(mu0, std0, mu1, std1):

    result = numpy.log(std1/(std0 + 0.000001))
    result+= ((std0**2) + (mu0 - mu1)**2)/(2*(std1**2))
    result+= -0.5

    print(mu0, std0, mu1, std1, round(result, 5))

    return result

kl_divergence(10, 1.5, 10, 1.5)

kl_divergence(0, 1, 1, 1)
kl_divergence(1, 1, 0, 1)
kl_divergence(1, 100, 0, 100)
kl_divergence(0, 100, 1, 100)

kl_divergence(0, 1, 0, 10)
kl_divergence(0, 10, 0, 1)
