# Welcome to your CDK TypeScript project

This is a blank project for CDK development with TypeScript.

The `cdk.json` file tells the CDK Toolkit how to execute your app.

## Useful commands

* `npm run build`   compile typescript to js
* `npm run watch`   watch for changes and compile
* `npm run test`    perform the jest unit tests
* `npx cdk deploy`  deploy this stack to your default AWS account/region
* `npx cdk diff`    compare deployed stack with current state
* `npx cdk synth`   emits the synthesized CloudFormation template

# Commands to test image on wsl
cd into image directory
1. docker build -t docker-image:test .
2. Set AWS Access Keys & AWS Secret IDs to your env variables.
3. docker run -p 9000:8080 -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} docker-image:test
4. (open seperate window)
5. curl -vvv -H "Content-Type: application/json" -d "{}" http://localhost:9000/2015-03-31/functions/function/invocations


# Deployment Commands
in root directory
1. aws sts get-caller-identity
2. cdk bootstrap --region us-east-1
3. cdk deploy
