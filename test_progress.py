import time

a = [1,2,3,4,5,6,7,8,9,10]

bar_length = 30
for i in a:
	progress = (i/10)
	sym = '='*int(progress * bar_length)

	if progress < 1:
		progress_string = '[' + sym + '>' + '-' * (bar_length- len(sym)-1) + ']'
	else:
		progress_string = '[' + sym + '-' * (bar_length- len(sym)) + ']'
	print(progress_string,end='\r')
	time.sleep(1)
print()