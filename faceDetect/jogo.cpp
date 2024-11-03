#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <SFML/Audio.hpp>
#include <limits>
#include <string>
#include <cstring>
#include <fstream>
#include <cctype>
#include <algorithm>
#include <ctype.h>

using namespace std;
using namespace cv;

int score[100];

void AbrirArquivo() {
    string linha;
    ifstream meu_arquivo("score.txt");
    if (meu_arquivo.is_open()) {
        int i = 0;
        while (getline(meu_arquivo, linha)) {
            score[i] = stoi(linha);
            i++;
        }
    }else {
        cout << "Incapaz de abrir o Arquivo" << endl;
    }
}

void SalvarArquivo() {
    sort(score, score + 100, greater<int>());
    ofstream meu_arquivo;
    meu_arquivo.open("score.txt");
    for (int i = 0; i < 100; i++) {
        if (score[i] == 0) {
            break;
        }
        meu_arquivo << score[i] << endl;
    }
     meu_arquivo.close();
}


void drawNoTransparency(Mat frame, Mat img, int xPos, int yPos){
    img.copyTo(frame.rowRange(yPos, yPos + img.rows).colRange(xPos, xPos + img.cols));
}

void circularDraw(Mat background, Mat frame, int &posicao){
    if (posicao > background.cols)
    { // verifica se o corte passou do tamanho do background
        posicao = 0;
    }

    // fazer dois cortes no background e colá-los em frame de forma a exibir algo que esteja circular
    Rect crop1(posicao, 0, background.cols - posicao, background.rows);
    Mat bg1 = background(crop1);
    Rect crop2(0, 0, posicao, background.rows);
    Mat bg2 = background(crop2);

    if (bg1.cols > 0)
    { // verifica se o primeiro corte tem algum tamanho
        bg1.copyTo(frame.rowRange(0, bg1.rows).colRange(0, bg1.cols));
    }
    if (bg2.cols > 1)
    { // verifica se o segundo corte tem algum tamanho
        bg2.copyTo(frame(Rect(frame.cols - posicao, 0, bg2.cols, bg2.rows)));
    }
}

void drawTransparency(Mat frame, Mat transp, int xPos, int yPos){
    Mat transp_copy = transp.clone(); // Criar uma cópia de transp
    Mat mask;
    vector<Mat> layers;

    // Verificar se a imagem possui 4 canais (incluindo o canal alfa)
    if (transp_copy.channels() == 4)
    {
        split(transp_copy, layers); // Separar os canais
        Mat rgb[3] = {layers[0], layers[1], layers[2]};
        mask = layers[3];           // Usar o canal alfa como máscara
        merge(rgb, 3, transp_copy); // Mesclar os canais RGB
        // Copiar a imagem com transparência para o frame usando a máscara alfa
        transp_copy.copyTo(frame.rowRange(yPos, yPos + transp_copy.rows).colRange(xPos, xPos + transp_copy.cols), mask);
    }
    else
    {
        // Se a imagem não possui 4 canais, não há transparência para aplicar
        // Neste caso, apenas copie a imagem para o frame
        drawNoTransparency(frame, transp_copy, xPos, yPos);
    }
}

// função para detectar faces e retornar sua posição
Rect detectFace(Mat &img, CascadeClassifier &cascade, double scale, bool tryflip){
    Mat gray, smallImg;
    vector<Rect> faces;

    double fx = 1 / scale;
    resize(img, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT);
    if (tryflip)
        flip(smallImg, smallImg, 1);
    cvtColor(smallImg, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    cascade.detectMultiScale(gray, faces,
                             1.3, 2, 0 | CASCADE_SCALE_IMAGE,
                             Size(40, 40));

    // Retorna apenas o primeiro retângulo de face detectado (se houver)
    if (!faces.empty())
    {
        return faces[0];
    }
    else
    {
        // Se nenhum rosto for detectado, retorne um retângulo vazio
        return Rect();
    }
}

string cascadeName;

char heightcompare(int faceposX, int faceposY, int midY){
    if (faceposX >= 0 && faceposX <= 1152)
    {
        if (faceposY > midY)
        {
            return 'p';
        }
        else
        {
            return 'o';
        }
        return 'o';
    }
    return 'o';
}

int main(){
    AbrirArquivo();
    // Som do jogo
    sf::SoundBuffer buffer;
    if (!buffer.loadFromFile("trilha.wav"))
    {
        cout << "Erro ao abrir aúdio" << endl;
        return 1;
    }
    // Criar um objeto de som e associá-lo ao buffer de áudio carregado
    sf::Sound sound;
    sound.setBuffer(buffer);
    sound.setLoop(true);
    sound.play();
    // neste caso não usaremos a camera como background e sim uma imagem
    // para ser o background, a qual irá ficar circulando (estilo flappy bird)

    /*Detecção de Face*/
    VideoCapture capture;
    Mat frame;
    bool tryflip;
    CascadeClassifier cascade;
    double scale;

    cascadeName = "haarcascade_frontalface_default.xml";
    scale = 2; // usar 1, 2, 4.
    if (scale < 1)
        scale = 1;
    tryflip = true;

    if (!cascade.load(cascadeName))
    {
        cerr << "ERROR: Could not load classifier cascade: " << cascadeName << endl;
        return -1;
    }

    if (!capture.open("video.mp4")) // para testar com um video
    // if(!capture.open(0)) // para testar com a webcam
    {
        cout << "Capture from camera #0 didn't work" << endl;
        return 1;
    }

    if (capture.isOpened())
    {
        cout << "Video capturing has been started ..." << endl;
    }

    Mat praia[5]; // definindo as imagens que iremos usar como background
    praia[0] = imread("praia_nascer.png", IMREAD_UNCHANGED);
    praia[1] = imread("praia_dia.png", IMREAD_UNCHANGED);
    praia[2] = imread("praia_meio_dia.png", IMREAD_UNCHANGED);
    praia[3] = imread("praia_tarde.png", IMREAD_UNCHANGED);
    praia[4] = imread("praia_noite.png", IMREAD_UNCHANGED);

    Mat robozin[8]; // definindo a matriz de imagens para o personagem
    robozin[0] = imread("robozin_1.png", IMREAD_UNCHANGED);
    robozin[1] = imread("robozin_2.png", IMREAD_UNCHANGED);
    robozin[2] = imread("robozin_3.png", IMREAD_UNCHANGED);
    robozin[3] = imread("robozin_4.png", IMREAD_UNCHANGED);
    robozin[4] = imread("robozin_5.png", IMREAD_UNCHANGED);
    robozin[5] = imread("robozin_6.png", IMREAD_UNCHANGED);
    robozin[6] = imread("robozin_pulando_1.png", IMREAD_UNCHANGED);
    robozin[7] = imread("robozin_pulando_2.png", IMREAD_UNCHANGED);

    Mat obstacle[3]; // definindo matrizes de imagens para obstáculos
    obstacle[0] = imread("obstacle_1.png", IMREAD_UNCHANGED);
    obstacle[1] = imread("obstacle_2.png", IMREAD_UNCHANGED);
    obstacle[2] = imread("obstacle_3.png", IMREAD_UNCHANGED);

    for (int i = 0; i < 5; i++)
    {
        cvtColor(praia[i], praia[i], COLOR_RGBA2RGB);
    }

    // temos que as imagens de background e obstáculos são muito grande, vamos dar um resize
    for (int i = 0; i < 5; i++)
    {
        resize(praia[i], praia[i], Size(), 1 / 2.0, 1 / 2.0, INTER_LINEAR_EXACT);
    }

    for (int i = 0; i < 3; i++)
    {
        resize(obstacle[i], obstacle[i], Size(), 1 / 1.6, 1 / 1.6, INTER_LINEAR_EXACT);
    }

    Mat background[6]; // criando uma matriz de matrizes para representar o background

    for (int i = 0; i < 5; i++)
    { // e atribuindo a cada uma imaegm que usaremos como background
        background[i] = praia[i].clone();
    }

    int velocidade = 10, posicao = 0; // a velocidade que se deseja fazer o background se mexer e
                                      // a variavel auxiliar posicao para o funcionamento da função
    do{
        cout << "|=================================|" << endl;
        cout << "|              Menu               |" << endl;
        cout << "|=================================|" << endl; 
        cout << "| --> Jogar = 1                   |" << endl;
        cout << "| --> Pontuacao = 2               |" << endl;
        cout << "|=================================|" << endl;
        int resp, resp1 = 0;
        do{
        cout << "--> Escolha a sua opcao: ";
        cin >> resp;
        switch (resp){
        case 1:
            resp1 = 1;
            break;
        case 2:
            for (int c = 0; c < 100; c++){
                if (score[c] == 0)
                {
                    continue;
                }
                cout << "Posicao " << c + 1 << " - " << score[c] << endl;
            }
            break;
        default:
        cout << "Opcao invalida!" << endl;
            break;
        }
        }while(resp1 != 1);

        int ipraia = 0, ipulo = 9;
        int irobo = 0, pulo = 0;
        int ap[10] = {0, 20, 30, 40, 50, 60, 70, 80, 90, 100};
        int apareci = 0;

        cout << "Canais praia debug " << praia[0].channels() << endl;
        cout << "Canais robozin debug " << robozin[0].channels() << endl;
        cout << "Canis background debug " << background[0].channels() << endl;

        int positionX = 1000;
        int auxctrl = 0;

        int scoretotal = 0;

        while (1)
        {

            capture >> frame;
            if (frame.empty())
                break;

            Rect face = detectFace(frame, cascade, scale, tryflip);

            // posição do centro da face
            int faceposX = face.x + (face.width) / 2;
            int faceposY = face.y + (face.height) / 2;

            // meio da tela
            int midY = (280);

            scoretotal += velocidade;
            Mat score(200, 400, CV_8UC3, Scalar(0, 0, 0));
            string score_str = to_string(scoretotal);

            if (ipraia == 5)
            { // Para retornar o background inicial
                ipraia = 0;
            }

            if (irobo == 6)
            { // Para retornar ao personagem inicial
                irobo = 0;
            }


            // Desenha o background circular
            circularDraw(praia[ipraia], background[ipraia], posicao);
            // circularDraw(praia[ipraia], obstacle[obstacle_aleatorio], posicao);
            posicao += velocidade;

            // desenhando uma imagem com transparencia no background
            if (pulo == 0)
            {
                if (ipulo < 9)
                {
                    drawTransparency(background[ipraia], robozin[7], 100, 400 + (ap[ipulo])); // desenhando a robô no chão
                    ipulo++;
                }
                else
                {
                    drawTransparency(background[ipraia], robozin[irobo], 100, 400 + (ap[ipulo]));
                }
            }
            else
            {
                drawTransparency(background[ipraia], robozin[6], 100, 500 - (ap[ipulo])); // desenhando a robô durante o pulo
                ipulo++;
                if (ipulo == 10)
                {
                    ipulo = 0;
                    pulo = 0;
                }
            }

            // Gerando um número aleatório
            int obstacle_aleatorio;
            if (auxctrl == 0)
            {
                // Gerando um número aleatório
                int numero_aleatorio = rand() % 100;

                // Gerando obstáculo
                if (numero_aleatorio < 5)
                {
                    auxctrl = 1;
                    obstacle_aleatorio = rand() % 3;
                }
            }
            else
            {
                drawTransparency(background[ipraia], obstacle[obstacle_aleatorio], positionX, 575); // desenhando o obstáculo no chão

                positionX -= 10;

                if (positionX < 50)
                {
                    positionX = 1000;
                    auxctrl = 0;
                }
            }

            if (heightcompare(faceposX, faceposY, midY) == 'p')
            {
                if (pulo == 0)
                {
                    pulo = 1;
                    ipulo = 0;
                }
            }

            char c = (char)waitKey(45);
            if (c == 27 || c == 'q')
            { // apertar q para sair do programa (quit)
                break;
            }
            else if (c == 'p')
            {
                if (pulo == 0)
                {
                    pulo = 1;
                    ipulo = 0;
                }
            }

            // cout << background[ipraia].cols << " x " << background[ipraia].rows << endl; // diz o tamanho da matriz do background
            putText(background[ipraia], "Score: " + score_str, Point(50, 100), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 139), 2);
            imshow("result", background[ipraia]); // abre uma aba exibindo uma matriz, o primeiro parametro é o nome da aba

            if (posicao % 1000 == 0) // if para manter uma imagem girando por um tempo antes de ir para próxima
                ipraia++;
            irobo++;
        }

        for (int c = 0; c < 100; c++){
            if (score[c] == 0){
                score[c] = scoretotal;
                break;
            }
        }

        cout << "|==================================|" << endl;
        cout << "|              Menu                |" << endl;
        cout << "|==================================|" << endl; 
        cout << "| --> Sair do jogo = 1             |" << endl;
        cout << "| --> Voltar ao menu principal = 2 |" << endl;
        cout << "|==================================|" << endl;
        int resp3 = 0, resp4 = 0;
        int saida = 0;
        do{
        cout << "--> Escolha a sua opcao: ";
        cin >> resp3;
        switch (resp3){
        case 1:
            resp4 = 1;
            saida = 1;
            break;
        case 2:
            resp4 = 1;
            saida = 0;
            break;
        default:
        cout << "Opcao invalida!" << endl;
            break;
        }
        }while(resp4 != 1);

        if (saida == 1){
            break;
        }

    } while (1);
    SalvarArquivo();
    return 0;
}