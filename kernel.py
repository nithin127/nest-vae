import numpy as np

def kernel(z, ktype='gaussian'):
	d = z.shape[0]
	K = np.zeros((d,d))
	for i in range(d):
		for j in range(i+1):
			if ktype == 'gaussian':
				K[i,j] = K[j,i] = np.exp(-0.5*(np.linalg.norm(z[i]-z[j])**2))
			elif ktype == 'laplacian':
				K[i,j] = K[j,i] = np.exp(-1.0*np.linalg.norm(z[i]-z[j]))
			else:
				raise('Kernel type unknown, please check the code')
	return K

def get_HSIC(z_array, ktype='gaussian'):
	# Assuming that the z_array.shape = [no.of examples, latent space dimension]
	m, d = z_array.shape 
	HSIC_array = np.zeros((d,d))
	H = np.eye(m) - 1.0/m
	for i in range(d):
		for j in range(i+1):
			K = kernel(z_array[:,i], ktype=ktype)
			L = kernel(z_array[:,j], ktype=ktype)
			HSIC_array[i,j] = HSIC_array[j,i] = np.trace\
				(np.matmul(np.matmul(np.matmul(K, H),L),H))/((m-1)**2)
	return HSIC_array

def main():
	x_mean = 0
	x_var = 10

	y_mean = 0
	y_var = 1

	z_mean = 10
	z_var = 10

	w_mean = 5
	w_var = 100

	x1 = np.random.normal(0,10,500)
	x2 = np.random.normal(0,1,500)
	x3 = np.random.normal(10,10,500)
	x4 = np.random.normal(5,100,500)
	x5 = np.random.normal(5,1,500)
	x6 = np.random.uniform(3,5,500)

	z_array = np.vstack((x1,x2,x3,x4,x5,x6)).T

	HSIC_array = get_HSIC(z_array)
	print('The mean value of the HSIC_array is {}'.format(np.mean(HSIC_array)))

if __name__ == "__main__":
	main()