from __future__ import print_function
from __future__ import division

import pickle
import bz2
from Plots import compute_kernel_alltimes
import sys
import os

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
		weights = []; log_update_probs = []
		for particle in particles:
			weights.append(particle.weight)
			log_update_probs.append(particle.log_update_prob)
		weights = np.array(weights)
		log_update_probs = np.array(log_update_probs)
		log_update_probs = log_update_probs - np.max(log_update_probs) # Prevents overflow
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

def readObservations(dataFile, outputFolder):
	observations = []
	wdToIndex, index = {}, 0
	with open(dataFile, "r", encoding="utf-8") as f:
		for i, line in enumerate(f):
			l = line.replace("\n", "").split("\t")
			timestamp = float(l[0])
			words = l[1].split(",")
			uniquewords, cntwords = np.unique(words, return_counts=True)
			for un in uniquewords:
				if un not in wdToIndex:
					wdToIndex[un] = index
					index += 1
			uniquewords = [wdToIndex[un] for un in uniquewords]
			uniquewords, cntwords = np.array(uniquewords, dtype=int), np.array(cntwords, dtype=int)

			tup = (i, timestamp, (uniquewords, cntwords))
			observations.append(tup)
	with open(outputFolder+"indexWords.txt", "w+", encoding="utf-8") as f:
		for wd in wdToIndex:
			f.write(f"{wdToIndex[wd]}\t{wd}\n")
	V = len(wdToIndex)
	return observations, V

def getLikTxt(cluster, theta0=None):
	cls_word_distribution = np.array(cluster.word_distribution, dtype=int)
	cls_word_count = int(cluster.word_count)

	vocabulary_size = len(cls_word_distribution)
	if theta0 is None:
		theta0 = 0.01

	priors_sum = theta0*vocabulary_size  # ATTENTION SEULEMENT SI THETA0 EST SYMMETRIQUE !!!!
	log_prob = 0

	cnt = np.bincount(cls_word_distribution)
	un = np.arange(len(cnt))

	log_prob += gammaln(priors_sum)
	log_prob += gammaln(cls_word_count+1)
	log_prob += gammaln(un + theta0).dot(cnt)  # ATTENTION SEULEMENT SI THETA0 EST SYMMETRIQUE !!!!

	log_prob -= gammaln(cls_word_count + priors_sum)
	log_prob -= vocabulary_size*gammaln(theta0)
	log_prob -= gammaln(cls_word_count+1)

	return log_prob

def writeParticles(DHP, folderOut, nameOut):
	with open(folderOut+nameOut+"_particles.txt", "w+") as f:
		for pIter, p in enumerate(DHP.particles):
			f.write(f"Particle\t{pIter}\t{p.weight}\t{p.docs2cluster_ID}\n")
			for c in p.clusters:
				likTxt = getLikTxt(p.clusters[c], theta0 = DHP.theta0[0])
				f.write(f"Cluster\t{c}\t{DHP.alpha0}\t{p.clusters[c].alpha}\t{likTxt}\t{p.clusters[c].word_count}\t[")
				V = len(p.clusters[c].word_distribution)
				for iwdd, wdd in enumerate(p.clusters[c].word_distribution):
					f.write(str(wdd))
					if iwdd != V:
						f.write(" ")
					else:
						f.write("]")
				f.write("\n")

def run_fit(observations, folderOut, nameOut, lamb0, means, sigs, r=1., theta0=None, alpha0 = None, sample_num=2000, particle_num=8, printRes=False, vocabulary_size=None):
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

	if vocabulary_size is None:
		allWds = set()
		for a in observations:
			for w in a[2][0]:
				allWds.add(w)
		vocabulary_size = len(list(allWds))+2
	if theta0 is None: theta0 = 1.
	if alpha0 is None: alpha0 = 1.

	particle_num = particle_num
	base_intensity = lamb0
	reference_time = means
	bandwidth = sigs
	theta0 = np.array([theta0 for _ in range(vocabulary_size)])
	alpha0 = np.array([alpha0] * len(means))
	sample_num = sample_num
	threshold = 1.0 / (particle_num*2.)

	DHP = Dirichlet_Hawkes_Process(particle_num = particle_num, base_intensity = base_intensity, theta0 = theta0,
								   alpha0 = alpha0, reference_time = reference_time, vocabulary_size = vocabulary_size,
								   bandwidth = bandwidth, sample_num = sample_num, r=r)

	t = time.time()

	lgObs = len(observations)
	for i, news_item in enumerate(observations):
		doc = parse_newsitem_2_doc(news_item = news_item, vocabulary_size = vocabulary_size)
		DHP.sequential_monte_carlo(doc, threshold)

		if i%100==1 and printRes:
			print(f'r={r} - Handling document {i}/{lgObs} (t={np.round(news_item[1]-observations[0][1], 1)}h) - Average time : {np.round((time.time()-t)*1000/(i), 0)}ms - '
				  f'Remaining time : {np.round((time.time()-t)*(len(observations)-i)/(i*3600), 2)}h - '
				  f'ClusTot={DHP.particles[0].cluster_num_by_now} - ActiveClus = {len(DHP.particles[0].active_clusters)}')

		if i%1000==1:
			while True:
				try:
					writeParticles(DHP, folderOut, nameOut)
					break
				except Exception as e:
					print(i, e)
					time.sleep(10)
					continue


	while True:
		try:
			writeParticles(DHP, folderOut, nameOut)
		except Exception as e:
			time.sleep(10)
			print(e)
			continue

def getArgs(args):
	import re
	dataFile, kernelFile, outputFolder, r, nbRuns, theta0, alpha0, sample_num, particle_num, printRes = [None]*10
	for a in args:
		print(a)
		try: dataFile = re.findall("(?<=data_file=)(.*)(?=)", a)[0]
		except: pass
		try: kernelFile = re.findall("(?<=kernel_file=)(.*)(?=)", a)[0]
		except: pass
		try: outputFolder = re.findall("(?<=output_folder=)(.*)(?=)", a)[0]
		except: pass
		try: r = re.findall("(?<=r=)(.*)(?=)", a)[0]
		except: pass
		try: nbRuns = int(re.findall("(?<=runs=)(.*)(?=)", a)[0])
		except: pass
		try: theta0 = float(re.findall("(?<=theta0=)(.*)(?=)", a)[0])
		except: pass
		try: alpha0 = float(re.findall("(?<=alpha0=)(.*)(?=)", a)[0])
		except: pass
		try: sample_num = int(re.findall("(?<=number_samples=)(.*)(?=)", a)[0])
		except: pass
		try: particle_num = int(re.findall("(?<=number_particles=)(.*)(?=)", a)[0])
		except: pass
		try: printRes = bool(re.findall("(?<=print_progress=)(.*)(?=)", a)[0])
		except: pass

	if dataFile is None:
		sys.exit("Enter a valid value for data_file")
	if kernelFile is None:
		sys.exit("Enter a valid value for kernel_file")
	if outputFolder is None:
		sys.exit("Enter a valid value for output_folder")
	if r is None: print("r value not found; defaulted to 1"); r="1"
	if nbRuns is None: print("nbRuns value not found; defaulted to 1"); nbRuns=1
	if theta0 is None: print("theta0 value not found; defaulted to 0.01"); theta0=0.01
	if alpha0 is None: print("alpha0 value not found; defaulted to 0.5"); alpha0=0.5
	if sample_num is None: print("sample_num value not found; defaulted to 2000"); sample_num=2000
	if particle_num is None: print("particle_num value not found; defaulted to 8"); particle_num=8
	if printRes is None: print("printRes value not found; defaulted to True"); printRes=True

	with open(kernelFile, 'r') as f:
		i=0
		tabMeans, tabSigs = [], []
		for line in f:
			if line=="\n":
				i += 1
				continue
			if i==0:
				lamb0 = float(line.replace("\n", ""))
			if i==1:
				tabMeans.append(float(line.replace("\n", "")))
			if i==2:
				tabSigs.append(float(line.replace("\n", "")))

	curdir = os.curdir+"/"
	for folder in outputFolder.split("/"):
		if folder not in os.listdir(curdir) and folder != "":
			os.mkdir(curdir+folder+"/")
		curdir += folder+"/"

	if len(tabMeans)!=len(tabSigs):
		sys.exit("The means and standard deviation do not match. Please check the parameters file.\n"
				 "The values should be organized as follows:\n[lambda_0]\n\n[mean_1]\n[mean_2]\n...\n[mean_K]\n\n[sigma_1]\n[sigma_2]\n...\n[sigma_K]\n")
	means = np.array(tabMeans)
	sigs = np.array(tabSigs)

	rarr = []
	for rstr in r.split(","):
		rarr.append(float(rstr))
	return dataFile, outputFolder, means, sigs, lamb0, rarr, nbRuns, theta0, alpha0, sample_num, particle_num, printRes

with open("test/Reddit.txt", "w+") as fout:
	with open("test/Reddit_events.txt", "r") as f:
		for i, line in enumerate(f):
			l = line.replace("\n", "").split("\t")
			timestamp = float(l[2])
			words = l[3].split(",")
			fout.write(f"{l[2]}\t{l[3]}\n")
#pause()
if __name__ == '__main__':
	dataFile, outputFolder, means, sigs, lamb0, arrR, nbRuns, theta0, alpha0, sample_num, particle_num, printRes = getArgs(sys.argv)

	observations, V = readObservations(dataFile, outputFolder)


	t = time.time()
	i = 0
	nbRunsTot = nbRuns*len(arrR)
	for run in range(nbRuns):
		for r in arrR:
			name = f"{dataFile[dataFile.rfind('/'):]}_r={r}_theta0={theta0}_alpha0={alpha0}_samplenum={sample_num}_particlenum={particle_num}_run_{run}"
			run_fit(observations, outputFolder, name, lamb0, means, sigs, r=r, theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num, printRes=printRes, vocabulary_size=V)
			print(f"r={r} - RUN {run}/{nbRuns} COMPLETE - REMAINING TIME: {np.round((time.time()-t)*(nbRunsTot-i)/(i*3600), 2)}h - ELAPSED TIME: {np.round((time.time()-t)/(3600), 2)}h")
			i += 1

