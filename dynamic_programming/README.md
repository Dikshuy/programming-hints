"""
Problem: Baxter Scott owns The Enlightened Dairy Co., a dairy company with magical cows. Early each morning, he brushes his teeth, strolls outside,
and finds that the cows have doubled in number. With double the number of cows, he can produce double the quantity of milk. While he is
ecstatic that he has so many cows and so much milk, the Dairy Regulator forces him to keep at most C cows on any given farm, which greatly
complicates his business. 
At The Enlightened Dairy Co., Baxter has access to an unlimited number of farms, each with a maximum capacity of
C cows. On each farm, cows reproduce at the same rate: they always double in number when the clock strikes midnight. To stay within the
Regulator’s rules, whenever a farm has strictly more than C cows, Baxter selects half of the cows on that farm and moves them to an 
entirely new, empty farm. More precisely, if there are D≤C cows on a farm, he leaves all D cows on the farm, but if there are D>C cows 
on a farm, he leaves ⌈D2⌉ cows on the farm and takes ⌊D2⌋ cows to a new, empty farm. (Here ⌈ ⌉ and ⌊ ⌋ denote the ceiling and floor 
functions, which round up/down to the nearest integer, respectively.) He does this early every morning, before the Regulator could possibly 
show up, so that he can avoid paying hefty Moo Fees.
The Regulator needs to know how many farms she will be inspecting when she visits The Enlightened Dairy Co. The Regulator inspects every 
farm that has at least one cow, and does not inspect any farm with zero cows. Given the number of cows on each farm with at least one cow 
on Day 0, compute the number of farms that need inspecting on any given day.
"""

def cows(C,N,M,initial_cows,dp,queries):
	cows_on_query = []
	# count the initial frequency of farms of different sizes
	for i in range(N):
		dp[0][initial_cows[i]] += 1

	for day in range(MAX_DAYS):
		# for all farm sizes double the number of cows
		for i in range(1,C+1):
			if i <= C/2:
				# cow count on farm with size "i" doubled, but number of farms didn't
				dp[day+1][2*i] += dp[day][i]
			else:
				# number of cows/farm on farm with size "i" exceeds the permitted limit, so double the number of farms
				dp[day+1][i] += 2*dp[day][i] 

	for i in queries:
		day = i
		cows_on_query.append(sum(dp[day]))
	
	return cows_on_query

if __name__ == '__main__':
	MAX_DAYS = 50
	C = 2
	N = 5
	M = 3
	initial_cows = [1,2,1,2,1]
	queries = [0,1,2]

	dp =[[0 for _ in range(C+1)] for _ in range(MAX_DAYS+1)]
	print(cows(C,N,M,initial_cows,dp,queries))

