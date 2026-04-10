import numpy as np
import albumentations as AT


def to_numpy(array):
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    return array    


class Scale:
    def __init__(self, scale, interpolation=1, p=1, is_testing=False):
        self.scale = scale
        self.interpolation = interpolation
        self.p = p
        self.is_testing = is_testing

    def __call__(self, image, mask=None):
        img = to_numpy(image)
        if not self.is_testing:
            msk = to_numpy(mask)

        imgh, imgw, _ = img.shape
        new_imgh, new_imgw = int(imgh * self.scale), int(imgw * self.scale)

        aug = AT.Resize(height=new_imgh, width=new_imgw, interpolation=self.interpolation, p=self.p)

        if self.is_testing:
            augmented = aug(image=img)
        else:
            augmented = aug(image=img, mask=msk)
        return augmented


class PadPairToMax(AT.DualTransform):
    def __init__(self, padding_value=0, mask_value=0, divisor=32, always_apply=False, p=1.0):
        super(PadPairToMax, self).__init__(always_apply, p)
        self.padding_value = padding_value
        self.mask_value = mask_value
        self.divisor = divisor

    def __call__(self, *args, force_apply=False, **kwargs):
        img1 = kwargs.get('image')
        img2 = kwargs.get('image2')
        
        h_max = 0
        w_max = 0
        
        if img1 is not None:
            img1 = to_numpy(img1)
            h, w = img1.shape[:2]
            h_max = max(h_max, h)
            w_max = max(w_max, w)
            
        if img2 is not None:
            img2 = to_numpy(img2)
            h, w = img2.shape[:2]
            h_max = max(h_max, h)
            w_max = max(w_max, w)
            
        if h_max == 0 or w_max == 0:
            return kwargs

        # Ensure dimensions are divisible by divisor
        if self.divisor > 1:
            h_max = int(np.ceil(h_max / self.divisor) * self.divisor)
            w_max = int(np.ceil(w_max / self.divisor) * self.divisor)

        def pad_image(img, h_target, w_target, val):
            if img is None: return None
            img = to_numpy(img)
            h, w = img.shape[:2]
            if h == h_target and w == w_target:
                return img
            
            pad_h = h_target - h
            pad_w = w_target - w
            
            # Pad bottom and right
            if len(img.shape) == 3:
                return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=val)
            else:
                return np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=val)

        new_kwargs = kwargs.copy()
        if img1 is not None:
            new_kwargs['image'] = pad_image(img1, h_max, w_max, self.padding_value)
        if img2 is not None:
            new_kwargs['image2'] = pad_image(img2, h_max, w_max, self.padding_value)
        if 'mask' in kwargs:
            new_kwargs['mask'] = pad_image(kwargs['mask'], h_max, w_max, self.mask_value)
            
        return new_kwargs
