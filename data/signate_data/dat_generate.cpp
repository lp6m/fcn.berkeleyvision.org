#include <iomanip>
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <glob.h>
using namespace std;
vector<string> get_file_path(string input_dir) {
    glob_t globbuf;
    vector<string> files;
    glob((input_dir + "*.png").c_str(), 0, NULL, &globbuf);
    for (int i = 0; i < globbuf.gl_pathc; i++) {
        files.push_back(globbuf.gl_pathv[i]);
    }
    globfree(&globbuf);
    return files;
}
using namespace std;
using namespace cv;
void process_image(string& imgpath, bool flip){
  cv::Mat img = cv::imread(imgpath);
  cout << imgpath << endl;
  cv::Mat resized_img = cv::Mat(227, 227, CV_8UC3);
  cout << img.rows << " " << img.cols << endl;
  resize(img, resized_img, resized_img.size(), 0, 0, INTER_NEAREST);
  if(flip){
      cv::flip(resized_img, resized_img, 1);
  }
  int height = resized_img.rows;
  int width = resized_img.cols;
  cout << height * width << endl;
  unsigned char index_array[height][width];
  for(int y = 0; y < resized_img.rows; y++){
    for(int x = 0; x < resized_img.cols; x++){
      cv::Vec<unsigned char, 3> pix = resized_img.ptr<cv::Vec3b>(y)[x];
      int b = (int)pix[0];
      int g = (int)pix[1];
      int r = (int)pix[2];
      unsigned char index = 4; //8bit
      if(b == 255 && g == 0 && r == 0){
        index = 3;//car
      }else if(b == 142 && g == 47 && r == 69){
        index = 0;//road
      }else if(b == 0 && g == 0 && r == 255){
        index = 1;//pedestrian
      }else if(b == 0 && g == 255 & r == 255){
        index = 2;//signal
      }
      index_array[y][x] = index;
    }
  }
  string basename = imgpath.substr(imgpath.find_last_of('/') + 1);
  basename = basename.substr(0, basename.find("."));
  string filename = "./seg_train_dat/" + basename + (flip ? "_flip" : "") + ".dat";
  cout << filename << endl;
  FILE* fp = fopen(filename.c_str(), "wb");
  fwrite(index_array, sizeof(unsigned char), height * width, fp);
  fclose(fp);
}
int main(void){
  vector<string> files = get_file_path("./seg_train_annotations/");
  for(auto file: files){
      process_image(file, false);
      process_image(file, true);
  }
}