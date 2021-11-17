# this shell script finds all agents trained for 5000 epochs starting from the current repository
# passed as the 1st arguments and copies them to the target repository (passed as 2nd argument)
# under keeping the underlying folder structure
find $1 -name '*agent_5000.msh' | cpio -pdm $2
find $1 -name '*normalizer_5000.msh' | cpio -pdm $2
 
