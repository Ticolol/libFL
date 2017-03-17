//
// Created by deangeli on 3/13/17.
//

#ifndef _HISTOGRAM_H_
#define _HISTOGRAM_H_

#include "image.h"
#include "common.h"
#include "adjacencyRelation.h"

typedef struct _histogram {
    float *val;
    int    n;
} Histogram;


Histogram *GrayHistogram(GrayImage *grayImage);
Histogram *GrayHistogram(GrayImage *grayImage, int nbins);
Histogram *GrayHistogram(GrayImage *grayImage, int maxValueAllowed, int minValueAllowed);
Histogram *GrayHistogram(GrayImage *grayImage, int nbins, int maxValueAllowed, int minValueAllowed);
Histogram *GrayHistogramFrom8BitGrayImage(GrayImage *grayImage);
Histogram *ColorHistogram(ColorImage *colorImage, int nbins);
Histogram *ColorHistogramFrom8bitColorIMage(ColorImage *colorImage, int nbinsPerChannel);
/*
 * computa a densidade de probabilidade para cada pixel da imagem baseado em todos os outros pixels
 * */
GrayImage *ProbabilityDensityFunction(ColorImage *img, double stdev);
GrayImage *ProbabilityDensityFunction(GrayImage *img, double stdev);

/*
 * computa a densidade de probabilidade para cada pixel da imagem baseado em sua adjacência
 * */
GrayImage *ProbabilityDensityFunction(ColorImage *img, double stdev,AdjacencyRelation *adjRel);
GrayImage *ProbabilityDensityFunction(GrayImage *img, double stdev,AdjacencyRelation *adjRel);

void WriteHistogram(Histogram *hist, char *filename);
void DestroyHistogram(Histogram **hist);




#endif //_HISTOGRAM_H
