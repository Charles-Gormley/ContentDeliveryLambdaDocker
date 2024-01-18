import os
import logging

# In image directory
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(processName)s] [%(levelname)s] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# building image
logging.info("Building Docker...")
os.system("docker build -t docker-image:test .")
logging.info("Docker Build Finished!")

# Deploying to aws
logging.info("Starting AWS Caller identity verification")
os.chdir("..")
os.system("aws sts get-caller-identity")
logging.info("Finished AWS Caller Identity verification")

logging.info("Starting CDK Bootstrap")
os.system("cdk bootstrap --region us-east-1")
logging.info("Finished CDK Bootstrap")

logging.info("Starting CDK Deployment")
os.system("cdk deploy")
logging.info("Finished Deployment :)")