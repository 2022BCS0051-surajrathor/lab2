pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "suraj4466/wine_predict_2022bcs0051"
    }

    stages {

        stage('Checkout') {
    steps {
        git branch: 'main',
        credentialsId: 'git-creds',
        url: 'https://github.com/2022BCS0051-surajrathor/lab2.git'
    }
}

        stage('Setup Python Virtual Environment') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                . venv/bin/activate
                python scripts/train.py
                '''
            }
        }

        stage('Read Accuracy') {
            steps {
                script {
                    ACCURACY = sh(
                        script: "jq .accuracy app/artifacts/metrics.json",
                        returnStdout: true
                    ).trim()

                    env.CURRENT_ACCURACY = ACCURACY
                    echo "Accuracy: ${ACCURACY}"
                }
            }
        }

        stage('Compare Accuracy') {
            steps {
                script {
                    withCredentials([string(credentialsId: 'best-accuracy', variable: 'BEST')]) {

                        if (CURRENT_ACCURACY.toFloat() > BEST.toFloat()) {
                            env.BUILD_MODEL = "true"
                            echo "New model is better"
                        } else {
                            env.BUILD_MODEL = "false"
                            echo "Model did not improve"
                        }
                    }
                }
            }
        }

        stage('Build Docker Image') {
            when {
                expression { env.BUILD_MODEL == "true" }
            }

            steps {
                sh '''
                docker build -t suraj4466/wine_predict_2022bcs0051:${BUILD_NUMBER} .
                docker tag suraj4466/wine_predict_2022bcs0051:${BUILD_NUMBER} suraj4466/wine_predict_2022bcs0051:latest
                '''
            }
        }

        stage('Push Docker Image') {
            when {
                expression { env.BUILD_MODEL == "true" }
            }

            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-creds',
                usernameVariable: 'USER',
                passwordVariable: 'PASS')]) {

                    sh '''
                    echo $PASS | docker login -u $USER --password-stdin
                    docker push suraj4466/wine_predict_2022bcs0051:${BUILD_NUMBER}
                    docker push suraj4466/wine_predict_2022bcs0051:latest
                    '''
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'app/artifacts/**'
        }
    }
}
