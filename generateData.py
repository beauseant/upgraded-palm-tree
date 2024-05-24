import src.sshiba as sshiba
import argparse

if __name__ == "__main__":

    # Parse args
    parser = argparse.ArgumentParser(description="Options for create shyntehtic data")

    parser.add_argument('-p', '--path', help='Output directory', required=True)
    parser.add_argument('-s', '--samples', help='Number of samples',default=1000, required=False)
    parser.add_argument('-i', '--inputFeatures', help='Number of input features', default=55, required=False)
    parser.add_argument('-f', '--outputFeatures', help='Number of output features', default=3, required=False)
    parser.add_argument('-c', '--commonLatent', help='Number of common latent variables', default=2, required=False)
    parser.add_argument('-l', '--firstViewLatent', help='Number of first view latent variables', default=3, required=False)
    parser.add_argument('-t', '--secondViewLatent', help='Number of second view latent variables',default=3, required=False)
    parser.add_argument('-k', '--myKc', help='Number of myKc', default=20, required=False)

    args = parser.parse_args()

    # Generate data
    data = sshiba.shinteticData()
    data.generateData(samples=int(args.samples), inputFeatures=int(args.inputFeatures),
                     outputFeatures=int(args.outputFeatures), commonLatent=int(args.commonLatent),
                      firstViewLatent=int(args.firstViewLatent), secondViewLatent=int(args.secondViewLatent), myKc=int(args.myKc)
                    )
    data.saveData(args.path)
