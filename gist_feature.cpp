#include "gist_feature.hpp"

using namespace retina::feature;

DEFINE_GLOBAL_FEATURE(GistFeature);

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

namespace {

/**
  sysmetric pad an array for FFT computation.
  @param	pA		source matrix data
  @param	Ma		width
  @param	Na		height
  @param	size	pad size
  @return   padded arrary
*/
double* sympadarray(double* pA, int Ma, int Na, int size) {
    int w = Ma + 2 * size;
    int h = Na + 2 * size;
    int i, j;
    int *idx = Malloc(int, w);
    double *ppA = Malloc(double, w * h);
    //memset(ppA,0,w*h*sizeof(double));
    for (i = 0; i < size; i++) {
        idx[i] = size - i - 1;
        idx[i + size + Ma] = Ma - 1 - i;
    }
    for (i = 0; i < Ma; i++) idx[i + size] = i;
    for (i = 0; i < w; i++)
        for (j = 0; j < h; j++)
            ppA[i * h + j] = pA[idx[i] * Na + idx[j]]; //transpose, due to matlab->array  ppA[(j+size)*w+i+size] = *ptr++;
    free(idx);
    return ppA;
}

/**
  calculate the blockwise mean.
  @param	data	source matrix data
  @param	m		width
  @param	n		height
  @param	v		output mean vector
  @param	nb		nb x nb grid
*/
void blockMean(double* data, int m, int n, double* v, int nb) {
    int i, j, x, y, s, xskip, yskip;
    xskip = m / nb;
    yskip = n / nb;
    memset(v, 0, sizeof(double) *nb * nb);
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) {
            x = i / xskip;
            y = j / yskip;
            v[x * nb + y] += data[i * n + j];
        }
    s = m * n / (nb * nb);
    for (i = 0; i < nb * nb; i++) v[i] /= (double) s;

}

/**
  shift for the fft result.
  @param	src	    source matrix for fftshift
  @param	m		width
  @param	n		height
  @return   shifted matrix
*/
double* fftshift(double* src, int m, int n) {
    int i, j, ii, jj;
    int p1 = m / 2, p2 = n / 2;
    double* dst = Malloc(double, m * n);
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) {
            if (i >= p1)
                ii = i - p1;
            else
                ii = i + p1;
            if (j >= p1)
                jj = j - p2;
            else
                jj = j + p2;
            dst[i * n + j] = src[ii * n + jj];
        }
    return dst;
}

/**
  generate GIST Gabor filter bank.
  @param	mask_size  64 or 128;Positive The size of the Gabor filter masks that it will generate
			nscale	   4, number of scales
			ori		   8, can be assigned for each scale
			nbf		   return total number of fitlers
  @return   gabor filters
*/
double* getGISTFilter(int mask_size, int nscale, int* ori, int* nbf) {
    int i, j, k;
    int nsize = mask_size * mask_size;
    int nfilters;
    double  tr, tf;
    double* param;
    double* shiftmask;
    double* t;
    double* mask = Malloc(double, nsize);
    double* t1   = Malloc(double, nsize);
    double* mG;
    int hafsize = mask_size / 2;
    nfilters = 0;
    for (i = 0; i < nscale; i++) 	nfilters += ori[i];

    *nbf  = nfilters;

    mG    = Malloc(double, nsize * nfilters);
    param = Malloc(double, nfilters * 4);

    k = 0;

    for (i = 0; i < nscale; i++)
        for (j = 0; j < ori[i]; j++) {
            param[k * 4]   = 0.35;
            param[k * 4 + 1] =  0.3 / (pow(1.85, (double) i));
            param[k * 4 + 2] = 16 * ori[i] * ori[i] / (32 * 32);
            param[k * 4 + 3] = M_PI / ori[i] * j;
            k++;
        }

    for (i = -hafsize; i < hafsize; i++)
        for (j = -hafsize; j < hafsize; j++)
            mask[(i + hafsize) *mask_size + j + hafsize] = sqrt((double)(i * i + j * j));

    shiftmask = fftshift(mask, mask_size, mask_size);

    for (i = -hafsize; i < hafsize; i++)
        for (j = -hafsize; j < hafsize; j++)
            t1[(i + hafsize) *mask_size + j + hafsize] = atan2((double) i, (double) j);

    t = fftshift(t1, mask_size, mask_size);

    for (i = 0; i < nfilters; i++) {

        for (j = 0; j < mask_size; j++)
            for (k = 0; k < mask_size; k++) {
                tr = t[j * mask_size + k] + param[i * 4 + 3];
                if (tr < -M_PI) tr += 2 * M_PI;
                if (tr > M_PI)  tr -= 2 * M_PI;
                tf = shiftmask[j * mask_size + k] / mask_size / param[i * 4 + 1] - 1.0;
                mG[i * nsize + j * mask_size + k] = exp(-10.0 * param[i * 4] * tf * tf - 2.*param[i * 4 + 2] * M_PI * tr * tr);
            }
    }

    free(param);
    free(mask);
    free(shiftmask);
    free(t);
    free(t1);

    return mG;
}

/**
  pre-process the image.
  @param	img			input image, also the output image
  @param	width		image width
  @param	height		image height
  @param	fc			filter parameter
*/
void prefilt(double* img, int width, int height, int fc) {
    int     i, j, ii, jj, n, w = 5;
    int     mask_size = width + 2 * w;
    double  x, y, s1 = (double) fc / sqrt(log(2.0));
    double* padim;
    double* gf;
    double* mask  = Malloc(double, mask_size * mask_size);
    CvMat* imdft  = cvCreateMatHeader(mask_size, mask_size, CV_64FC2);
    cvCreateData(imdft);

    // Pad images to reduce boundary artifacts
    for (i = 0; i < width * height; i++)	img[i] = log(img[i] + 1);

    padim = sympadarray(img, width, height, w);

    n = std::max(width + 2 * w, height + 2 * w);
    n = n + n % 2;
    // Filter
    for (i = -n / 2; i < n / 2; i++)
        for (j = -n / 2; j < n / 2; j++)
            mask[(i + n / 2) *n + j + n / 2] = exp(- (double)(i * i + j * j) / (s1 * s1));

    gf = fftshift(mask, n, n);
    //Whitening 	output = img - real(ifft2(fft2(img).*gf));
    for (i = 0; i < mask_size; i++)
        for (j = 0; j < mask_size; j++) {
            ((double*)(imdft->data.ptr + imdft->step * i)) [j * 2] = padim[i * mask_size + j];
            ((double*)(imdft->data.ptr + imdft->step * i)) [j * 2 + 1] = 0.0;
        }
    cvDFT(imdft, imdft, CV_DXT_FORWARD, 0);

    for (i = 0; i < mask_size; i++)
        for (j = 0; j < mask_size; j++) {
            ((double*)(imdft->data.ptr + imdft->step * i)) [j * 2]   *= gf[i * mask_size + j];
            ((double*)(imdft->data.ptr + imdft->step * i)) [j * 2 + 1] *= gf[i * mask_size + j];
        }

    cvDFT(imdft, imdft, CV_DXT_INV_SCALE, mask_size);
    // Local contrast normalization localstd = repmat(sqrt(abs(ifft2(fft2(mean(output,3).^2).*gf(:,:,1,:)))), [1 1 c 1]);
    for (i = 0; i < mask_size; i++)
        for (j = 0; j < mask_size; j++) {
            padim[i * mask_size + j] -= ((double*)(imdft->data.ptr + imdft->step * i)) [j * 2];
            ((double*)(imdft->data.ptr + imdft->step * i)) [j * 2]    = padim[i * mask_size + j] * padim[i * mask_size + j];
            ((double*)(imdft->data.ptr + imdft->step * i)) [j * 2 + 1] = 0.0;
        }

    cvDFT(imdft, imdft, CV_DXT_FORWARD, 0);
    for (i = 0; i < mask_size; i++)
        for (j = 0; j < mask_size; j++) {
            ((double*)(imdft->data.ptr + imdft->step * i)) [j * 2]   *= gf[i * mask_size + j];
            ((double*)(imdft->data.ptr + imdft->step * i)) [j * 2 + 1] *= gf[i * mask_size + j];
        }

    cvDFT(imdft, imdft, CV_DXT_INV_SCALE, mask_size);

    //output = output./(.2+localstd); Crop output to have same size than the input, output = output(w+1:sn-w, w+1:sm-w,:,:);
    for (i = 0; i < width; i++)
        for (j = 0; j < height; j++) {
            ii = i + w;
            jj = j + w;
            x = ((double*)(imdft->data.ptr + imdft->step * ii)) [jj * 2];
            y = ((double*)(imdft->data.ptr + imdft->step * ii)) [jj * 2 + 1];

            img[i * height + j] = padim[ii * mask_size + jj] / (0.2 + sqrt(sqrt(x * x + y * y)));
        }

    free(mask);
    free(gf);
    free(padim);
    cvReleaseMat(&imdft);
}

/**
  extract gist feature.
  g = Mag(IFFT(FFT(IM)*GroupFilter{i})), and then extract the blockwise mean.
  try blockwise-moment later
  @param	mG		Gabor fiter bank
  @param	nbf		total number of Gabor filters
  @param	mask_size	mask size of the gabor filter, also the image is resize to mask size
  @param	img		img matrix
  @param	nblocks	number of blocks to extract the grid mean
  @return   extracted GIST features.
*/
double* gist(const double* mG, int nbf, int mask_size, double* img, int nblocks) {
    int i, j, c;
    int W = nblocks * nblocks;
    int nsize = mask_size * mask_size;
    double u, v;
    const double* mGf;

    double* g    = Malloc(double, W * nbf);

    double* igo  = Malloc(double, nsize);
    CvMat* imdft = cvCreateMatHeader(mask_size, mask_size, CV_64FC2);
    CvMat* igfft = cvCreateMatHeader(mask_size, mask_size, CV_64FC2);

    //allocate memory
    cvCreateData(imdft);
    cvCreateData(igfft);

    for (i = 0; i < mask_size; i++)
        for (j = 0; j < mask_size; j++) {
            ((double*)(imdft->data.ptr + imdft->step * i)) [j * 2]   = img[i * mask_size + j];
            ((double*)(imdft->data.ptr + imdft->step * i)) [j * 2 + 1] = 0.0;
        }

    cvDFT(imdft, imdft, CV_DXT_FORWARD, 0);

    for (c = 0; c < nbf; c++) {
        mGf = mG + c * nsize;
        //	ig = abs(ifft2(img.*repmat(G(:,:,n), [1 1 N])));
        for (i = 0; i < mask_size; i++)
            for (j = 0; j < mask_size; j++) {
                ((double*)(igfft->data.ptr + igfft->step * i)) [j * 2]   = ((double*)(imdft->data.ptr + imdft->step * i)) [j * 2] * mGf[i * mask_size + j];
                ((double*)(igfft->data.ptr + igfft->step * i)) [j * 2 + 1] = ((double*)(imdft->data.ptr + imdft->step * i)) [j * 2 + 1] * mGf[i * mask_size + j];
            }

        cvDFT(igfft, igfft, CV_DXT_INV_SCALE, mask_size);

        for (i = 0; i < mask_size; i++)
            for (j = 0; j < mask_size; j++) {
                u = ((double*)(igfft->data.ptr + igfft->step * i)) [j * 2];
                v = ((double*)(igfft->data.ptr + igfft->step * i)) [j * 2 + 1];
                igo[i * mask_size + j] = sqrt(u * u + v * v);
            }
        //compute blockwise mean
        blockMean(igo, mask_size, mask_size, g + c * W, nblocks);
    }
    free(igo);
    cvReleaseMat(&imdft);
    cvReleaseMat(&igfft);

    return g;
}

/**
  Convert IplImage to the unsigned char array.
  @param    im          IplImage pointor.
  @param    arr         unsigned char arrary
*/
double* iplToDArr(IplImage* img) {
    int i, j;
    int w = img->width, h = img->height;
    double* arr = Malloc(double, w * h);
    if (img->nChannels == 3) {
        for (j = 0; j < h; j++)
            for (i = 0; i < w; i++)
                arr[j * w + i] = (((uchar*)(img->imageData + img->widthStep * j))[i * 3] +
                                  ((uchar*)(img->imageData + img->widthStep * j))[i * 3 + 1] +
                                  ((uchar*)(img->imageData + img->widthStep * j))[i * 3 + 2]) / 3.0;
    } else {
        for (j = 0; j < h; j++)
            for (i = 0; i < w; i++)
                arr[j * w + i] = ((uchar*)(img->imageData + img->widthStep * j))[i];
    }
    return arr;
}


struct GistParameter {
    constexpr static int gist_scale = 4, gist_size = 256, gist_nb = 4, gist_fc = 4;
    static int nbf;
    static int gist_ori[4];
    static int gist_dim;
    static double* mG;
    GistParameter() {
        // create gabor filer bank to extract GIST feature
        mG = getGISTFilter(gist_size, gist_scale, gist_ori, &nbf);
        gist_dim = nbf * gist_nb * gist_nb;
    }
    ~GistParameter() {
        free(mG);
    }
} __gist_parameter;
int GistParameter::gist_ori[4] = {8, 8, 8, 8};
double* GistParameter::mG = 0;
int GistParameter::gist_dim = 0;
int GistParameter::nbf = 0;

}

GistFeature GistFeature::extract(const cv::Mat& image) {
    cv::Mat image8;
    if (image.depth() != 8) image8 = image;
    else image.convertTo(image8, CV_8U);

    cv::Mat gist_image;
    cv::resize(image8, gist_image, cv::Size(GistParameter::gist_size, GistParameter::gist_size));

    IplImage gist_image_ipl = gist_image;
    double* gist_pixel_array = iplToDArr(&gist_image_ipl);
    prefilt(gist_pixel_array, GistParameter::gist_size, GistParameter::gist_size, GistParameter::gist_fc);

    double* gist_vector = gist(GistParameter::mG, GistParameter::nbf, GistParameter::gist_size, gist_pixel_array, GistParameter::gist_nb);

    // write back
    GistFeature feature;
    for (int i = 0; i < DIM; i++)
        feature[i] = 60000.0 * gist_vector[i];

    free(gist_pixel_array);
    free(gist_vector);

    return feature;
}

GistFeature GistFeature::extract(const cv::Mat& image, const cv::Mat& mask) {
//    throw NotSupportedException();
}
