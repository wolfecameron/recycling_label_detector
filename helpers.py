"""Contains all helper methods used for parsing recycling number from an image"""


def find_similar_contours(cont_tups, num_cnts):
	"""Finds num_cnts contours that have area that is
	similar (threshold of 1.1) - these correspond to very
	similar contours that constitute three parts of a shape

	Returns none if similar contours not found	
	
	Returns first such contour group that is found
	
	Parameters:
	cont_tups-- list of tuples of form (area, contour)
	num_cnts-- number of contours to be found for the result
	"""
	
	l_ind = 0
	for r_ind in range(num_cnts, len(cont_tups) + 1):
		# check to see if this group of contours is the same
		sub_list = cont_tups[l_ind:r_ind]
		if(same_cnts(sub_list)):
			# return contours that are the same
			return [cnt[1] for cnt in sub_list]
		l_ind += 1
	
	return None

def same_cnts(cnt_tups):
	"""returns True if the contours all have area within 1.1
	ratio in value to the contour to the left of it
	"""
	
	# any ratio greater than this not considered same
	sim_ratio = 1.1
	
	# check to see if all area values are within threshold
	for r_ind in range(1, len(cnt_tups)):
		r_area = cnt_tups[r_ind][0]
		l_area = cnt_tups[r_ind - 1][0]
		if(r_area/l_area > sim_ratio):
			return False
	
	# return true if all values within the similarity threshold
	return True
	
