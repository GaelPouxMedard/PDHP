from __future__ import print_function
from __future__ import division

import datetime
import pickle
import bz2

import numpy as np

from utils import *
import copyreg as copy_reg
import types
from copy import deepcopy as copy
import time

from Evaluation import compDists, confMat

np.random.seed(1111)

def _pickle_method(m):
	if m.im_self is None:
		return getattr, (m.im_class, m.im_func.func_name)
	else:
		return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


def g(x):
	numerator = - (x - 0) ** 2 / (2 * 0.5 ** 2)
	denominator = (2 * np.pi * 0.5 ** 2 ) ** 0.5
	return np.exp(numerator) / denominator

arr = []
import matplotlib.pyplot as plt
for x in np.linspace(0, 1, 100):
	v = 4*g(x)/3 - g(1-x)/3
	arr.append(v)
plt.plot(np.linspace(0, 1, 100), arr)
#plt.show()
#pause()

class Dirichlet_Hawkes_Process(object):
	"""docstring for Dirichlet Hawkes Prcess"""
	def __init__(self, particle_num, base_intensity, theta0, alpha0, reference_time, vocabulary_size, bandwidth, sample_num, r):
		super(Dirichlet_Hawkes_Process, self).__init__()
		self.r = r
		self.particle_num = particle_num
		self.base_intensity = base_intensity
		self.theta0 = theta0
		self.alpha0 = alpha0
		self.reference_time = reference_time
		self.vocabulary_size = vocabulary_size
		self.bandwidth = bandwidth
		self.horizon = (np.max(self.reference_time)+3*np.max(self.bandwidth))*2
		self.sample_num = sample_num
		self.particles = []
		for i in range(particle_num):
			self.particles.append(Particle(weight = 1.0 / self.particle_num))

		self.active_interval = None


	def sequential_monte_carlo(self, doc, threshold):
		# Set relevant time interval
		tu = EfficientImplementation(doc.timestamp, self.reference_time, self.bandwidth)
		T = doc.timestamp + self.horizon  # So that Gaussian RBF kernel is fully computed; needed to correctly compute the integral part of the likelihood
		self.active_interval = [tu, T]

		particles = []
		for particle in self.particles:
			particles.append(self.particle_sampler(particle, doc))

		self.particles = particles

		# Resample particules whose weight is below the given threshold
		self.particles = self.particles_normal_resampling(self.particles, threshold)

	def particle_sampler(self, particle, doc):
		# Sample cluster label
		particle, selected_cluster_index = self.sampling_cluster_label(particle, doc)
		# Update the triggering kernel
		particle.clusters[selected_cluster_index].alpha = self.parameter_estimation(particle, selected_cluster_index)
		# Calculate the weight update probability
		particle.log_update_prob = self.calculate_particle_log_update_prob(particle, selected_cluster_index, doc)
		return particle

	def sampling_cluster_label(self, particle, doc):
		if len(particle.clusters) == 0: # The first document is observed
			particle.cluster_num_by_now += 1
			selected_cluster_index = particle.cluster_num_by_now
			selected_cluster = Cluster(index = selected_cluster_index, num_samples=self.sample_num, alpha0=self.alpha0)
			selected_cluster.add_document(doc)
			particle.clusters[selected_cluster_index] = selected_cluster #.append(selected_cluster)
			particle.docs2cluster_ID.append(selected_cluster_index)
			particle.active_clusters[selected_cluster_index] = [doc.timestamp]
			self.active_cluster_logrates = {0:0, 1:0}

		else: # A new document arrives
			active_cluster_indexes = [0] # Zero for new cluster
			active_cluster_rates = [self.base_intensity**self.r]
			cls0_log_dirichlet_multinomial_distribution = log_dirichlet_multinomial_distribution(doc.word_distribution, doc.word_distribution,\
			 doc.word_count, doc.word_count, self.vocabulary_size, self.theta0)
			active_cluster_textual_probs = [cls0_log_dirichlet_multinomial_distribution]
			# Update list of relevant timestamps
			particle.active_clusters = self.update_active_clusters(particle)

			# Posterior probability for each cluster
			for active_cluster_index in particle.active_clusters:
				timeseq = particle.active_clusters[active_cluster_index]
				active_cluster_indexes.append(active_cluster_index)
				time_intervals = doc.timestamp - np.array(timeseq)
				alpha = particle.clusters[active_cluster_index].alpha
				rate = triggering_kernel(alpha, self.reference_time, time_intervals, self.bandwidth)

				# Powered Dirichlet-Hawkes prior
				active_cluster_rates.append(rate)

				# Language model likelihood
				cls_word_distribution = particle.clusters[active_cluster_index].word_distribution + doc.word_distribution
				cls_word_count = particle.clusters[active_cluster_index].word_count + doc.word_count
				cls_log_dirichlet_multinomial_distribution = log_dirichlet_multinomial_distribution(cls_word_distribution, doc.word_distribution,\
				 cls_word_count, doc.word_count, self.vocabulary_size, self.theta0)
				active_cluster_textual_probs.append(cls_log_dirichlet_multinomial_distribution)

			# Posteriors to probabilities
			active_cluster_logrates = self.r*np.log(np.array(active_cluster_rates)+1e-100)
			self.active_cluster_logrates = {c: active_cluster_logrates[i+1] for i, c in enumerate(particle.active_clusters)}
			self.active_cluster_logrates[0] = active_cluster_logrates[0]
			cluster_selection_probs = active_cluster_logrates + active_cluster_textual_probs # in log scale
			cluster_selection_probs = cluster_selection_probs - np.max(cluster_selection_probs) # prevent overflow
			cluster_selection_probs = np.exp(cluster_selection_probs)
			cluster_selection_probs = cluster_selection_probs / np.sum(cluster_selection_probs)

			# Random cluster selection
			selected_cluster_array = multinomial(exp_num = 1, probabilities = cluster_selection_probs)
			selected_cluster_index = np.array(active_cluster_indexes)[np.nonzero(selected_cluster_array)][0]

			# New cluster drawn
			if selected_cluster_index == 0:
				particle.cluster_num_by_now += 1
				selected_cluster_index = particle.cluster_num_by_now
				self.active_cluster_logrates[selected_cluster_index] = self.active_cluster_logrates[0]
				selected_cluster = Cluster(index = selected_cluster_index, num_samples=self.sample_num, alpha0=self.alpha0)
				selected_cluster.add_document(doc)
				particle.clusters[selected_cluster_index] = selected_cluster
				particle.docs2cluster_ID.append(selected_cluster_index)
				particle.active_clusters[selected_cluster_index] = [doc.timestamp]

			# Existing cluster drawn
			else:
				selected_cluster = particle.clusters[selected_cluster_index]
				selected_cluster.add_document(doc)
				particle.docs2cluster_ID.append(selected_cluster_index)
				particle.active_clusters[selected_cluster_index].append(doc.timestamp)

		return particle, selected_cluster_index

	def parameter_estimation(self, particle, selected_cluster_index):
		timeseq = np.array(particle.active_clusters[selected_cluster_index])

		# Observation is alone in the cluster => the cluster is new => random initialization of alpha
		# Note that it cannot be a previously filled cluster since it would have 0 chance to get selected (see sampling_cluster_label)
		if len(timeseq)==1:
			alpha = dirichlet(self.alpha0)
			return alpha

		T = self.active_interval[1]
		particle.clusters[selected_cluster_index] = update_cluster_likelihoods(timeseq, particle.clusters[selected_cluster_index], self.reference_time, self.bandwidth, self.base_intensity, T)
		alpha = update_triggering_kernel_optim(particle.clusters[selected_cluster_index])
		return alpha

	def update_active_clusters(self, particle):
		tu = self.active_interval[0]
		keys = list(particle.active_clusters.keys())
		for cluster_index in keys:
			timeseq = particle.active_clusters[cluster_index]
			active_timeseq = [t for t in timeseq if t > tu]
			if not active_timeseq:
				del particle.active_clusters[cluster_index]  # If no observation is relevant anymore, the cluster has 0 chance to get chosen => we remove it from the calculations
				del particle.clusters[cluster_index].alphas
				del particle.clusters[cluster_index].log_priors
				del particle.clusters[cluster_index].likelihood_samples
				del particle.clusters[cluster_index].triggers
				del particle.clusters[cluster_index].integ_triggers
			else:
				particle.active_clusters[cluster_index] = active_timeseq
		return particle.active_clusters
	
	def calculate_particle_log_update_prob(self, particle, selected_cluster_index, doc):
		cls_word_distribution = particle.clusters[selected_cluster_index].word_distribution
		cls_word_count = particle.clusters[selected_cluster_index].word_count
		doc_word_distribution = doc.word_distribution
		doc_word_count = doc.word_count

		log_update_prob = log_dirichlet_multinomial_distribution(cls_word_distribution, doc_word_distribution, cls_word_count, doc_word_count, self.vocabulary_size, self.theta0)

		lograte = np.exp(self.active_cluster_logrates[selected_cluster_index])
		lograte = lograte / np.sum(np.exp(list(self.active_cluster_logrates.values())))

		log_update_prob += lograte

		return log_update_prob

	def particles_normal_resampling(self, particles, threshold):
		#print('\nparticles_normal_resampling')
		weights = []; log_update_probs = []
		for particle in particles:
			weights.append(particle.weight)
			log_update_probs.append(particle.log_update_prob)
		weights = np.array(weights)
		log_update_probs = np.array(log_update_probs)
		log_update_probs = log_update_probs - np.max(log_update_probs) # prevent overflow
		update_probs = np.exp(log_update_probs)
		weights = weights * update_probs
		weights = weights / np.sum(weights) # normalization
		resample_num = len(np.where(weights + 1e-5 < threshold)[0])

		if resample_num == 0: # No need to resample particle, but still need to assign the updated weights to particles
			for i, particle in enumerate(particles):
				particle.weight = weights[i]
			return particles
		else:
			remaining_particles = [particle for i, particle in enumerate(particles) if weights[i] + 1e-5 > threshold ]
			resample_probs = weights[np.where(weights + 1e-5 > threshold)]
			resample_probs = resample_probs/np.sum(resample_probs)
			remaining_particle_weights = weights[np.where(weights + 1e-5 > threshold)]
			for i,_ in enumerate(remaining_particles):
				remaining_particles[i].weight = remaining_particle_weights[i]

			resample_distribution = multinomial(exp_num = resample_num, probabilities = resample_probs)
			if not resample_distribution.shape: # The case of only one particle left
				for _ in range(resample_num):
					new_particle = copy(remaining_particles[0])
					remaining_particles.append(new_particle)
			else: # The case of more than one particle left
				for i, resample_times in enumerate(resample_distribution):
					for _ in range(resample_times):
						new_particle = copy(remaining_particles[i])
						remaining_particles.append(new_particle)

			# Normalize the particle weights
			update_weights = np.array([particle.weight for particle in remaining_particles]); update_weights = update_weights / np.sum(update_weights)
			for i, particle in enumerate(remaining_particles):
				particle.weight = update_weights[i]

			self.particles = None
			return remaining_particles

def fitTick(observations, means, sigs, alphaTrue):
	from tick.hawkes import HawkesEM
	from tick.plot import plot_hawkes_kernels
	import matplotlib.pyplot as plt
	em = HawkesEM(15, kernel_size=30, n_threads=8, verbose=False, tol=1e-3)
	ts = np.array(np.array(observations, dtype=object)[:, 1], dtype=float)[:1000]
	em.fit([np.ascontiguousarray(ts)])
	plot_hawkes_kernels(em, show=False)
	arr = np.linspace(0, np.max(means+np.max(sigs)), 1000)
	try:
		plt.plot(arr, triggering_kernel(alphaTrue[0,0], means, arr, sigs, donotsum=True), label="True kernel 1")
		plt.plot(arr, triggering_kernel(alphaTrue[1,1], means, arr, sigs, donotsum=True), label="True kernel 2")
	except:
		plt.plot(arr, triggering_kernel(alphaTrue, means, arr, sigs, donotsum=True), label="True kernel 1")

	plt.legend()
	plt.show()

def parse_newsitem_2_doc(news_item, vocabulary_size):
	''' convert (id, timestamp, word_distribution, word_count) to the form of document
	'''
	#print(news_item)
	index = news_item[0]
	timestamp = news_item[1] # / 3600.0 # unix time in hour
	word_id = news_item [2][0]
	count = news_item[2][1]
	word_distribution = np.zeros(vocabulary_size)
	word_distribution[word_id] = count
	word_count = np.sum(count)
	doc = Document(index, timestamp, word_distribution, word_count)
	# assert doc.word_count == np.sum(doc.word_distribution)
	return doc

def readData(folder, name):
	observations = []
	with open(folder+name+"_events.txt", "r") as f:
		for i, line in enumerate(f):
			l = line.replace("\n", "").split("\t")
			clusTemp = int(float(l[0]))
			clusTxt = int(float(l[1]))
			timestamp = float(l[2])
			words = l[3].split(",")
			uniquewords, cntwords = np.unique(words, return_counts=True)
			uniquewords, cntwords = np.array(uniquewords, dtype=int), np.array(cntwords, dtype=int)

			tup = (i, timestamp, (uniquewords, cntwords), clusTemp, clusTxt)
			observations.append(tup)
	with open(folder+name+"_lamb0.txt", "r") as f:
		lamb0 = float(f.read().replace("\n", ""))

	means = np.loadtxt(folder+name+"_means.txt")
	sigs = np.loadtxt(folder+name+"_sigs.txt")
	try:
		alpha = np.load(folder+name+"_alpha.npy")
	except:
		alpha=None

	return observations, lamb0, means, sigs, alpha

def plotSynthData(observations):
	import matplotlib.pyplot as plt
	colors = ["r", "b", "y", "g", "orange", "cyan","purple"]
	for e in observations:
		c = int(e[-1])
		t = e[1]
		plt.plot(t, -1/10-c/10, "o", c=colors[c], markersize=4)
	plt.show(block=False)
	plt.pause(1)

def writeParticles(DHP, folderOut, nameOut):
	from Plots import getLikTxt
	with open(folderOut+nameOut+"_particles.txt", "w+") as f:
		for pIter, p in enumerate(DHP.particles):
			f.write(f"Particle\t{pIter}\t{p.weight}\t{p.docs2cluster_ID}\n")
			for c in p.clusters:
				likTxt = getLikTxt(p.clusters[c], theta0 = DHP.theta0[0])
				f.write(f"Cluster\t{c}\t{DHP.alpha0}\t{p.clusters[c].alpha}\t{likTxt}\t{p.clusters[c].word_count}\n")

def run_fit_synth(params):
	r, folder, folderOut, nbClasses, run_time, perc_rand, overlap_temp, overlap_voc, words_per_obs, voc_per_class, run, runDS = params
	name = f"Obs_nbclasses={nbClasses}_lg={run_time}_overlapvoc={overlap_voc}_overlaptemp={overlap_temp}_percrandomizedclus={perc_rand}_vocperclass={voc_per_class}_wordsperevent={words_per_obs}_run={run}"
	nameOut = f"Obs_nbclasses={nbClasses}_lg={run_time}_overlapvoc={overlap_voc}_overlaptemp={overlap_temp}_r={r}_percrandomizedclus={perc_rand}_vocperclass={voc_per_class}_wordsperevent={words_per_obs}_run={run}_runDS={runDS}"

	observations, lamb0, means, sigs, alpha = readData(folder, name)

	run_fit(observations, folderOut, nameOut, lamb0, means, sigs, r=r, alphaTrue=alpha, sample_num=2000, particle_num=8, printRes=True)

def run_fit_rw(params):
	r, folder, folderOut, runDS, XP = params
	if "Reddit" in folder:
		name = XP
		nameOut = f"{XP}_{r}"
	else:
		print("PROBLEEEEEEEEEEEEM")
		name, nameOut = "", ""
		pause()

	observations, lamb0, means, sigs, alpha = readData(folder, name)

	run_fit(observations, folderOut, nameOut, lamb0, means, sigs, r=r, alphaTrue=None, theta0=0.01, alpha0 = 0.5, sample_num=2000, particle_num=8, printRes=True) #t0=0.01

def run_fit(observations, folderOut, nameOut, lamb0, means, sigs, r=1., theta0=1., alpha0 = None, sample_num=2000, particle_num=8, printRes=False, alphaTrue=None):
	"""
	observations = ([array int] index_obs, [array float] timestamp, ([array int] unique_words, [array int] count_words), [opt, int] temporal_cluster, [opt, int] textual_cluster)
	folderOut = Output folder for the results
	nameOut = Name of the file to which _particles_compressed.pklbz2 will be added
	lamb0 = base intensity
	means, sigs = means and sigmas of the gaussian RBF kernel
	r = exponent parameter of the Powered Dirichlet process; defaults to 1. (standard Dirichlet process)
	theta0 = value of the language model symmetric Dirichlet prior
	alpha0 = symmetric Dirichlet prior from which samples used in Gibbs sampling are drawn (estimation of alpha)
	sample_num = number of samples used in Gibbs sampling
	particle_num = number of particles used in the Sequential Monte-Carlo algorithm
	printRes = whether to print the results according to ground-truth (optional parameters of observations and alpha)
	alphaTrue = ground truth alpha matrix used to generate the observations from gaussian RBF kernel
	"""

	particle_num = particle_num
	allWds = set()
	for a in observations:
		for w in a[2][0]:
			allWds.add(w)
	vocabulary_size = len(list(allWds))+2

	base_intensity = lamb0
	if theta0 is None: theta0 = 1.
	theta0 = np.array([theta0 for _ in range(vocabulary_size)])
	if alpha0 is None: alpha0 = 1.
	alpha0 = np.array([alpha0] * len(means))
	reference_time = means
	bandwidth = sigs
	sample_num = sample_num
	threshold = 1.0 / (particle_num*1.5)

	if alphaTrue is not None:
		un, cnt = np.unique(np.array(observations, dtype=list)[:, 4], return_counts=True)
		print(f"Classes: {un}, Counts: {cnt}")
		#plotSynthData(observations[:1000])


	DHP = Dirichlet_Hawkes_Process(particle_num = particle_num, base_intensity = base_intensity, theta0 = theta0,
								   alpha0 = alpha0, reference_time = reference_time, vocabulary_size = vocabulary_size,
								   bandwidth = bandwidth, sample_num = sample_num, r=r)

	t = time.time()

	nbToFit = -1

	#fitTick(observations, means, sigs, alphaTrue)
	for i, news_item in enumerate(observations[1:nbToFit]):
		doc = parse_newsitem_2_doc(news_item = news_item, vocabulary_size = vocabulary_size)
		DHP.sequential_monte_carlo(doc, threshold)

		if i%100==1 and printRes:
			print(f'r={r} - Handling document {i}/{len(observations)} - t={np.round(news_item[1]-observations[1][1], 1)}h - Average time : {np.round((time.time()-t)*1000/(i), 0)}ms - '
				  f'Remaining time : {np.round((time.time()-t)*(len(observations)-i)/(i*3600), 2)}h - '
				  f'ClusTot={DHP.particles[0].cluster_num_by_now} - ActiveClus = {len(DHP.particles[0].active_clusters)}')

			if alphaTrue is not None:
				un, cnt = np.unique(np.array(observations, dtype=object)[:i + 2, 4], return_counts=True)
				print(un, cnt, confMat(observations[:i + 2], DHP.particles)[0][0:2], compDists(observations[:i + 1], alphaTrue, [DHP.particles[0]], reference_time, bandwidth)[0])

			if i>300 and False:
				lgClus, selectedClusters = [], []
				for c in DHP.particles[0].clusters:
					inds = np.where(np.array(DHP.particles[0].docs2cluster_ID, dtype=int)==int(c))[0]
					selectedClusters.append(c)
					lgClus.append(len(inds))
				selectedClusters = [c for _, c in sorted(zip(lgClus, selectedClusters), reverse=True)]
				lgClus = [l for l, _ in sorted(zip(lgClus, selectedClusters), reverse=True)]
				selectedClusters = selectedClusters[:5]
				lgClus = lgClus[:5]
				for ind_c, c in enumerate(selectedClusters):
					print(lgClus[ind_c], selectedClusters[ind_c], list(DHP.particles[0].clusters[c].alpha)[:9])
				plt.close()
				alphas = np.array([dirichlet(DHP.alpha0) for _ in range(2000)])
				log_priors = np.array([log_dirichlet_PDF(a, DHP.alpha0) for a in alphas])
				print(means)
				for c in DHP.particles[0].clusters:
					inds = np.where(np.array(DHP.particles[0].docs2cluster_ID, dtype=int)==int(c))[0]
					a = np.array(observations, dtype=object)[inds, 1]
					a = np.array(a, dtype=float)
					if len(a)<28:
						continue
					print(len(a), DHP.particles[0].clusters[c].alpha)
					sa = np.subtract.outer(a, a)
					sa = sa[sa>0].flatten()
					plt.hist(sa, bins = 100, alpha=0.2, density=True)
					newalpha = update_triggering_kernel(a, alphas, DHP.reference_time, DHP.bandwidth, DHP.base_intensity, DHP.active_interval[1], log_priors, DHP.r)
					dt = np.linspace(0, np.max(sa), 1000)
					plt.plot(dt, triggering_kernel(DHP.particles[0].clusters[c].alpha, means, dt, sigs, donotsum=True), "-b")
					plt.plot(dt, triggering_kernel(newalpha, means, dt, sigs, donotsum=True), "-r")
					#plt.semilogy()
					plt.show()
					#fitTick(np.array(observations, dtype=object)[inds], means, sigs, newalpha)
					#plt.show()
					#print(newalpha[:5], DHP.particles[0].clusters[c].alpha[:5])
					a = a-np.min(a)
					from Plots import compute_kernel_alltimes
					x, y = compute_kernel_alltimes(a, means, sigs, DHP.particles[0].clusters[c].alpha, res=1000)
					dt = np.linspace(0, np.max(x), 1000)
					#plt.plot(dt, 5*c+triggering_kernel(DHP.particles[0].clusters[c].alpha, means, dt, sigs, donotsum=True), "-b")
					#plt.plot(x, 5*c+y, "-k")
					#plt.plot(a, [5*c]*len(a), "o", markersize=1)
				plt.show()
		if i%1000==1:
			while True:
				try:
					if "Reddit" in nameOut:
						writeParticles(DHP, folderOut, nameOut)
					else:
						with bz2.BZ2File(folderOut+nameOut+'_particles_compressed.pklbz2', 'w') as sfile:
							pickle.dump(DHP.particles, sfile)
					break
				except Exception as e:
					print(i, e)
					time.sleep(10)
					continue


	while True:
		try:
			if "Reddit" in nameOut:
				writeParticles(DHP, folderOut, nameOut)
			else:
				with bz2.BZ2File(folderOut+nameOut+'_particles_compressed.pklbz2', 'w') as sfile:
					pickle.dump(DHP.particles, sfile)
			break
		except Exception as e:
			time.sleep(10)
			print(e)
			continue

if __name__ == '__main__':
	import sys
	try:
		folderData = sys.argv[1]
		XP = sys.argv[2]
		firstRunNumber = int(sys.argv[3])
		nbRunsPerDS = int(sys.argv[4])
		words_per_obs = int(sys.argv[5])
		rUsr = sys.argv[6]
	except Exception as e:
		folderData = "Reddit"
		XP = "Reddit"
		folderData = "Reddit"
		XP = "Reddit_3"
		firstRunNumber = 0
		nbRunsPerDS = 1
		words_per_obs = -1
		rUsr = 0.5
		pass

	folder=f"data/{folderData}/"
	folderOut=f"output/{folderData}/"
	nbClasses = 2
	run_time = 1500
	perc_rand = 0.
	overlap_voc = 0.
	overlap_temp = 0.
	voc_per_class = 1000
	nbRuns = 0

	np.random.seed(1111 + firstRunNumber * 10)



	if False:
		arrR = [0., 0.4, 0.8, 1., 1.5, 2., 3., 5., 7.5, 10.]+[0.2, 0.6, 0.9, 1.1, 1.3, 2.5, 4., 6., 8.5]
		overlap_voc = 0.
		overlap_temp = 0.
		i=0
		t = time.time()
		firstRunNumber = 0
		words_per_obs = 20
		fstrun = 0
		nbRuns = 5
		runDS = 0
		nbExec = nbRunsPerDS*nbRuns*len(arrR)*11

		perc_rand = 0.
		nbExec = nbRunsPerDS*nbRuns*len(arrR)*len([0., 0.3, 0.5, 0.7])*len([0., 0.3, 0.5, 0.7, 0.9])
		for run in range(fstrun, fstrun+nbRuns):
			for r in arrR:
				for overlap_temp in [0., 0.3, 0.5, 0.7]:
					for overlap_voc in [0.5]:
						params = (r, folder, folderOut, nbClasses, run_time, perc_rand, overlap_temp, overlap_voc, words_per_obs, voc_per_class, run, runDS)
						print(f"runDS={runDS} - run={run} - r={r} - {nbClasses} classes - OL_text={overlap_voc} - OL_temp={overlap_temp}")
						run_fit_synth(params)

						i+=1
						print(f"Average time : {np.round((time.time()-t)/(i), 0)}s - Remaining time : {np.round((time.time()-t)*(nbExec-i)/(i*3600), 2)}h - Elapsed time : {np.round((time.time()-t)/3600, 2)}h\n")


		perc_rand = 0.
		nbExec = nbRunsPerDS*nbRuns*len(arrR)*len([0., 0.3, 0.5, 0.7])*len([0., 0.3, 0.5, 0.7, 0.9])
		for run in range(fstrun, fstrun+nbRuns):
			for r in arrR:
				for overlap_temp in [0.5]:
					for overlap_voc in [0., 0.3, 0.5, 0.7, 0.9]:
						params = (r, folder, folderOut, nbClasses, run_time, perc_rand, overlap_temp, overlap_voc, words_per_obs, voc_per_class, run, runDS)
						print(f"runDS={runDS} - run={run} - r={r} - {nbClasses} classes - OL_text={overlap_voc} - OL_temp={overlap_temp}")
						run_fit_synth(params)

						i+=1
						print(f"Average time : {np.round((time.time()-t)/(i), 0)}s - Remaining time : {np.round((time.time()-t)*(nbExec-i)/(i*3600), 2)}h - Elapsed time : {np.round((time.time()-t)/3600, 2)}h\n")


		print("End redone Synth", fstrun, nbRuns)
		pause()


	'''
		import pprofile
		profiler = pprofile.Profile()
		with profiler:
			# Single run
			run = 0
			runDS = 0
			r = 2.
			params = (r, folder, folderOut, nbClasses, run_time, perc_rand, overlap_temp, overlap_voc, words_per_obs, voc_per_class, run, runDS)
			print(f"r={r} - {nbClasses} classes - OL_text={overlap_voc} - OL_temp={overlap_temp} - run={run}")
			run_fit(params)
		profiler.dump_stats("Bench.txt")
		pause()
		'''

	import time
	t = time.time()
	i = 0

	if rUsr is None or rUsr=="r1":
		arrR = [0., 0.4, 0.8, 1., 1.5, 2., 3., 5., 7.5, 10.]
	if rUsr == "r2":
		arrR = [0.2, 0.6, 0.9, 1.1, 1.3, 2.5, 4., 6., 8.5]

	if XP=="Decorr":
		overlap_voc = 0.
		overlap_temp = 0.
		nbExec = nbRunsPerDS*nbRuns*len(arrR)*11
		for runDS in range(firstRunNumber, firstRunNumber+nbRunsPerDS):
			for run in range(nbRuns):
				for r in arrR:
					for perc_rand in np.array(list(range(11)))/10:
						params = (r, folder, folderOut, nbClasses, run_time, perc_rand, overlap_temp, overlap_voc, words_per_obs, voc_per_class, run, runDS)
						print(f"runDS={runDS} - run={run} - r={r} - {nbClasses} classes - OL_text={overlap_voc} - OL_temp={overlap_temp} - perc_rand={perc_rand}")
						run_fit_synth(params)

						i+=1
						print(f"Average time : {np.round((time.time()-t)/(i), 0)}s - Remaining time : {np.round((time.time()-t)*(nbExec-i)/(i*3600), 2)}h - Elapsed time : {np.round((time.time()-t)/3600, 2)}h\n")

	if XP=="Overlap":
		perc_rand = 0.
		nbExec = nbRunsPerDS*nbRuns*len(arrR)*len([0., 0.3, 0.5, 0.7])*len([0., 0.3, 0.5, 0.7, 0.9])
		for runDS in range(firstRunNumber, firstRunNumber+nbRunsPerDS):
			for run in range(nbRuns):
				for r in arrR:
					for overlap_temp in [0., 0.3, 0.5, 0.7]:
						for overlap_voc in [0., 0.3, 0.5, 0.7, 0.9]:
							params = (r, folder, folderOut, nbClasses, run_time, perc_rand, overlap_temp, overlap_voc, words_per_obs, voc_per_class, run, runDS)
							print(f"runDS={runDS} - run={run} - r={r} - {nbClasses} classes - OL_text={overlap_voc} - OL_temp={overlap_temp}")
							run_fit_synth(params)

							i+=1
							print(f"Average time : {np.round((time.time()-t)/(i), 0)}s - Remaining time : {np.round((time.time()-t)*(nbExec-i)/(i*3600), 2)}h - Elapsed time : {np.round((time.time()-t)/3600, 2)}h\n")

	if "Reddit" in XP:
		#arrR = [0.1, 1., 5., 10.]
		if rUsr is None:
			rReddit = 1.
		else:
			arrR = [float(rUsr)]
		nbExec = nbRunsPerDS*len(arrR)
		for runDS in range(firstRunNumber, firstRunNumber+nbRunsPerDS):
			for r in arrR:
				print(f"runDS={runDS} - r={r}")
				params = (r, folder, folderOut, runDS, XP)
				run_fit_rw(params)
				i += 1
				print(f"Average time : {np.round((time.time()-t)/(i), 0)}s - Remaining time : {np.round((time.time()-t)*(nbExec-i)/(i*3600), 2)}h - Elapsed time : {np.round((time.time()-t)/3600, 2)}h\n")


