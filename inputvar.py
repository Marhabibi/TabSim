import sys
import getopt

def inputvarsTrain(av,g):
	try:
		opts,args = getopt.getopt(av,"he:l:v:w:i:o:",["epoch=", "learning_rate=","embedding=","input_file=","output_dir="])
	except:
		print(g,"-e <epoch> -l <learning_rate> -v <embedding> -i <input_file> -o <output_dir>")
		sys.exit(2)
	if len(opts)<5:
		print(g,"-e <epoch> -l <learning_rate> -v <embedding> -i <input_file> -o <output_dir>")
		sys.exit(2)
	epoch = ""
	lr = ""
	out = ""
	
	inp = ""
	emb_vec = ""
	for opt,arg in opts:
		if opt == "-h":
			print(g,"-e <epoch> -l <learning_rate> -v <embedding> -i <input_file> -o <output_dir>")
			sys.exit(2)
		elif opt in ("-e","--epoch"):
			epoch = arg
		
		elif opt in ("-v","--embedding"):
			emb_vec = arg
		elif opt in ("-i","--input_file"):
			inp = arg
		elif opt in ("-o","--output_dir"):
			out = arg
		elif opt in("-l","--learning_rate"):
			lr = arg
	return epoch,out,lr,inp,emb_vec


def inputvarsEval(av,g):
	try:
		opts,args = getopt.getopt(av,"hm:i:o:",["model=","input=","output="])
	except:
		print(g,"-m <model> -i <input> -o <output>")
		sys.exit(2)
	outp = ""
	model = ""
	inp = ""
	if len(opts) < 3:
		print(g,"-m <model> -i <input> -o <output>")
		sys.exit(2)
	for opt,arg in opts:
		if opt == "-h":
			print(g,"-m <model> -i <input> -o <output>")
			sys.exit(2)
		elif opt in ("-o","--output"):
			outp = arg
		elif opt in ("-i","--input"):
			inp = arg
		elif opt in ("-m","--model"):
			model = arg

	return model,outp,inp

