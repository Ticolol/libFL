#include "histogram.h"
#include <omp.h>



Histogram *GrayHistogram(GrayImage *grayImage, int nbins)
{
    Histogram *hist = (Histogram *) calloc(1,sizeof(Histogram));
    int p, n = grayImage->ncols*grayImage->nrows;
    int i, Imax=MaximumValue(grayImage), Imin=MinimumValue(grayImage);
    int K = (Imax-Imin)/MIN(nbins,Imax-Imin); /* size of each interval */

    hist->n   = nbins+1;
    hist->val = (float *)calloc(hist->n,sizeof(float));

    for (p=0; p <  n; p++)
        hist->val[(grayImage->val[p]-Imin)/K]++;

    for (i=0; i <  hist->n; i++) {
        hist->val[i] /= n;
    }

    return(hist);
}


Histogram *GrayHistogram(GrayImage *grayImage, int maxValueAllowed, int minValueAllowed)
{
    Histogram *hist = (Histogram *) calloc(1,sizeof(Histogram));
    int p, n = grayImage->ncols*grayImage->nrows;
    int i;

    hist->n   = maxValueAllowed-minValueAllowed+1;
    hist->val = (float *)calloc(hist->n,sizeof(float));

    for (p=0; p <  n; p++)
        hist->val[grayImage->val[p]-minValueAllowed] += 1;

    for (i=0; i <  hist->n; i++)
        hist->val[i] /= n;

    return(hist);
}

Histogram *GrayHistogram(GrayImage *grayImage, int nbins, int maxValueAllowed, int minValueAllowed)
{
    Histogram *hist = (Histogram *) calloc(1,sizeof(Histogram));
    int p, n = grayImage->ncols*grayImage->nrows;
    int i;

    hist->n   = nbins;
    float step = (maxValueAllowed-minValueAllowed+1)/(float)nbins;
    hist->val = (float *)calloc(hist->n,sizeof(float));
    int binIndex = -1;

    for (p=0; p <  n; p++)
        binIndex = (grayImage->val[p]-minValueAllowed)/step;
    hist->val[binIndex] += 1.0f;

    for (i=0; i <  hist->n; i++)
        hist->val[i] /= n;

    return(hist);
}

Histogram *GrayHistogramFrom8BitGrayImage(GrayImage *grayImage){
    Histogram *hist = GrayHistogram(grayImage, 255, 0);
    return hist;
}

/* K = 2^b / w is the size of each interval, where b is the depth of
   the image per channel (we are assuming 8 bits per channel) and w is
   the number of intervals in each axis. */

Histogram *ColorHistogram(ColorImage *colorImage, int nbins)
{
    Histogram *hist = (Histogram *) calloc(1,sizeof(Histogram));
    int xp, yp, n = colorImage->nx*colorImage->ny;
    int i,  w = (pow((double)nbins,(1.0/3.0))+0.5), K = (MaximumColorValue(colorImage)+1) / w + 1;


    hist->n   = w*w*w + 1;
    hist->val = (float *)calloc(hist->n,sizeof(float));

    for (yp=0; yp < colorImage->ny; yp++){
        for (xp=0; xp < colorImage->nx; xp++){
            int l1, l2, l3;
            l1 = colorImage->cor[yp][xp].val[0]/K;
            l2 = colorImage->cor[yp][xp].val[1]/K;
            l3 = colorImage->cor[yp][xp].val[2]/K;
            i  = l1 + l2*w + l3*w*w;
            hist->val[i]++;
        }
    }

    for (i=0; i <  hist->n; i++) {
        hist->val[i] /= n;
    }


    return(hist);
}


Histogram *ColorHistogramFrom8bitColorIMage(ColorImage *colorImage, int nbinsPerChannel)
{
    Histogram *hist = (Histogram *) calloc(1,sizeof(Histogram));
    int xp, yp, n = colorImage->nx*colorImage->ny;
    int i,  w = nbinsPerChannel;
    float binSize = 256/(nbinsPerChannel);


    hist->n   = w*w*w ;
    hist->val = (float *)calloc(hist->n,sizeof(float));
    int l1, l2, l3;
    for (yp=0; yp < colorImage->ny; yp++){
        for (xp=0; xp < colorImage->nx; xp++){
            l1 = colorImage->cor[yp][xp].val[0]/binSize;
            l2 = colorImage->cor[yp][xp].val[1]/binSize;
            l3 = colorImage->cor[yp][xp].val[2]/binSize;
            i  = l1 + l2*w + l3*w*w;
            hist->val[i] += 1.0f;
        }
    }

    for (i=0; i <  hist->n; i++) {
        hist->val[i] /= n;
    }


    return(hist);
}


GrayImage *DensidadeProbabilidadeComDistanciaEuclidiana(ColorImage *img, float K)
{
    GrayImage *pdf = CreateGrayImage(img->nx, img->ny);
    int n = pdf->nrows*pdf->ncols;

#pragma omp parallel for
    for (int p=0; p < n; p++) {
        double val=0.0, Rp, Gp, Bp;
        Rp = img->cor[p/img->nx][p%img->nx].val[0]/255.0;
        Gp = img->cor[p/img->nx][p%img->nx].val[1]/255.0;
        Bp = img->cor[p/img->nx][p%img->nx].val[2]/255.0;

        for (int q=0; q < n; q++) {
            if (p != q) {
                double dist, Rq, Gq, Bq;
                Rq = img->cor[q/img->nx][q%img->nx].val[0]/255.0;
                Gq = img->cor[q/img->nx][q%img->nx].val[1]/255.0;
                Bq = img->cor[q/img->nx][q%img->nx].val[2]/255.0;
                dist = (Rp-Rq)*(Rp-Rq) + (Gp-Gq)*(Gp-Gq) + (Bp-Bq)*(Bp-Bq);
                dist = sqrt(dist);
                //if (dist < K) {
                val += exp(-dist);
                //}
            }
        }
        val /= (n-1);
        pdf->val[p] = (int)(255*val);
    }

    return(pdf);
}

void WriteHistogram(Histogram *hist, char *filename)
{
    FILE *fp = fopen(filename,"w");
    int   i;

    for (i=0; i < hist->n; i++)
        fprintf(fp,"%d %f\n",i,hist->val[i]);

    fclose(fp);
}

GrayImage *ProbabilityDensityFunction(ColorImage *img, double stdev)
{
    GrayImage *pdf = CreateGrayImage(img->nx, img->ny);
    int n = pdf->nrows*pdf->ncols;
    float  *val = (float *)calloc(img->nx*img->ny,sizeof(float));
    double  K = 2.0*stdev*stdev, maxdist = 3*stdev*3*stdev;
    float   Imax = MaximumColorValue(img), maxval, minval;

    printf("maxdist %f\n",maxdist);

    maxval = INT_MIN; minval = INT_MAX;
#pragma omp parallel for
    for (int p=0; p < n; p++) {
        double Rp, Gp, Bp;
        Rp = img->cor[p/img->nx][p%img->nx].val[0]/Imax;
        Gp = img->cor[p/img->nx][p%img->nx].val[1]/Imax;
        Bp = img->cor[p/img->nx][p%img->nx].val[2]/Imax;

        for (int q=0; q < n; q++) {
            if (p != q) {
                double dist, Rq, Gq, Bq;
                Rq = img->cor[q/img->nx][q%img->nx].val[0]/Imax;
                Gq = img->cor[q/img->nx][q%img->nx].val[1]/Imax;
                Bq = img->cor[q/img->nx][q%img->nx].val[2]/Imax;
                dist = (Rp-Rq)*(Rp-Rq) + (Gp-Gq)*(Gp-Gq) + (Bp-Bq)*(Bp-Bq);
                if (dist <= maxdist){
                    val[p] += exp(-dist/K);
                }
            }
        }
        if (val[p] > maxval)
            maxval = val[p];
        if (val[p] < minval)
            minval = val[p];
    }

    if (minval != maxval)
        for (int p=0; p < n; p++) {
            pdf->val[p] = (int)(65535*(val[p]-minval)/(maxval-minval));
        }
    free(val);

    return(pdf);
}

GrayImage *ProbabilityDensityFunction(GrayImage *img, double stdev)
{
    GrayImage *pdf = CreateGrayImage(img->ncols, img->nrows);
    int n = pdf->nrows*pdf->ncols;
    float  *val = (float *)calloc(img->ncols*img->nrows,sizeof(float));
    double  K = 2.0*stdev*stdev, maxdist = 3*stdev*3*stdev;
    float   Imax = MaximumValue(img), maxval, minval;

    printf("maxdist %f\n",maxdist);

    maxval = INT_MIN; minval = INT_MAX;
#pragma omp parallel for
    for (int p=0; p < n; p++) {
        double grayp;
        grayp = img->val[p]/Imax;


        for (int q=0; q < n; q++) {
            if (p != q) {
                double dist, grayq;
                grayq = img->val[q]/Imax;
                dist = (grayp-grayq);

                if (dist <= maxdist){
                    val[p] += exp(-dist/K);
                }
            }
        }
        if (val[p] > maxval)
            maxval = val[p];
        if (val[p] < minval)
            minval = val[p];

    }

    if (minval != maxval)
        for (int p=0; p < n; p++) {
            pdf->val[p] = (int)(65535*(val[p]-minval)/(maxval-minval));
        }
    free(val);

    return(pdf);
}

void DestroyHistogram(Histogram **hist)
{
    if (*hist != NULL) {
        free((*hist)->val);
        free(*hist);
    }
    *hist = NULL;
}



//int main(int argc, char **argv) {
//  char             ext[10],*pos;
//  Histogram       *hist=NULL;
//
//  if (argc != 3) {
//    printf("histogram <image.[pgm,ppm]> <histogram.txt>\n");
//    exit(-1);
//  }
//
//  pos = strrchr(argv[1],'.') + 1;
//  sscanf(pos,"%s",ext);
//
//  if (strcmp(ext,"pgm")==0){
//    GrayImage *grayImage = ReadGrayImage(argv[1]);
//
//    hist = GrayHistogram(grayImage);
//
//  } else { /* ppm */
//
//    ColorImage *colorImage =  ReadColorImage(argv[1]);
//
//    hist = ColorHistogram(colorImage,512);
//    GrayImage *pdf = DensidadeProbabilidade(colorImage,0.5);
//    WriteGrayImage(pdf,"densidade.pgm");
//    DestroyGrayImage(&pdf);
//  }
//
//
//  WriteHistogram(hist,argv[2]);
//  DestroyHistogram(&hist);
//
//  return 0;
//}
