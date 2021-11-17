# this shell script finds events files (i.e. tensorboard logs) starting from the current repository
# passed as the 1st arguments and copies them to the target repository (passed as 2nd argument)
# under keeping the underlying folder structure
find $1 -name '*events.*' | cpio -pdm $2
 
