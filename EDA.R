####MOST FREQUENT WORDS####
#Note to self: numpy arrays start at 0!!!

#top 50 words that appear most overall
wordcounts = apply(X, 2, sum)
sorted = sort(wordcounts, decreasing = TRUE)
all_indices = which(wordcounts %in% sorted[1:50]) 
#top 50 words are the first 50 except 51 instead of 49, mostly punctuation


#top 50 words that appear most in positive tweets
wordcounts_pos = apply(X[y == 1,], 2, sum)
sorted_pos = sort(wordcounts_pos, decreasing = TRUE)
pos_indices = which(wordcounts_pos %in% sorted_pos[1:50]) 
#1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 33
#34 35 36 37 38 39 40 41 42 43 44 47 48 51 53 54 59 93

#top 50 words that appear most in negative
wordcounts_neg = apply(X[y == 0,], 2, sum)
sorted_neg = sort(wordcounts_neg, decreasing = TRUE)
neg_indices = which(wordcounts_neg %in% sorted_neg[1:50]) 

#Most common words in negative tweets that aren't in the list of most common positive tweet words
neg_indices[!(neg_indices %in% pos_indices)] 
#not no go today do too work

#Most common words in positive tweets that aren't in the list of most common negative tweet words
pos_indices[!(pos_indices %in% neg_indices)] 
#good quot it's your love lol thanks







