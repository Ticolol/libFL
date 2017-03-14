#include "FL.h"

int main(int argc, char **argv) {
    char             ext[10],*pos;
    if (argc != 4) {
        printf("pdf <image.ppm> <stdev> <pdf.pgm>\n");
        exit(-1);
    }

    pos = strrchr(argv[1],'.') + 1;
    sscanf(pos,"%s",ext);

    if (strcmp(ext,"ppm")==0){
        ColorImage *colorImage =  ReadColorImage(argv[1]);
        //ColorImage *ycbcr      = RGBtoYCbCr(colorImage);
        //GrayImage  *pdf        = ProbabilityDensityFunction(ycbcr,atof(argv[2]));
        GrayImage  *pdf        = ProbabilityDensityFunction(colorImage,atof(argv[2]));
        WriteGrayImage(pdf,argv[3]);

        DestroyGrayImage(&pdf);
        DestroyColorImage(&colorImage);
        //DestroyColorImage(&ycbcr);
    }else{ /* ppm */
        GrayImage *grayImage =  ReadGrayImage(argv[1]);
        GrayImage  *pdf        = ProbabilityDensityFunction(grayImage,atof(argv[2]));
        WriteGrayImage(pdf,argv[3]);
        DestroyGrayImage(&pdf);
        DestroyGrayImage(&grayImage);
    }
    return 0;
}

