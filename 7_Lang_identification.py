"""
pip install langid

langid.py comes pre-trained on 97 languages (ISO 639-1 codes given):
af, am, an, ar, as, az, be, bg, bn, br, bs, ca, cs, cy, da, de, dz, el, en, eo, es, et, eu, fa, fi, fo, fr, ga, gl, gu, 
he, hi, hr, ht, hu, hy, id, is, it, ja, jv, ka, kk, km, kn, ko, ku, ky, la, lb, lo, lt, lv, mg, mk, ml, mn, mr, ms, mt, 
nb, ne, nl, nn, no, oc, or, pa, pl, ps, pt, qu, ro, ru, rw, se, si, sk, sl, sq, sr, sv, sw, ta, te, th, tl, tr, ug, uk, 
ur, vi, vo, wa, xh, zh, zu
You can also use langdetect (only 55 languages supported)
"""
import langid
print(langid.rank("Questa e una prova"))
print(langid.classify("Questa e una prova"))
print(langid.classify("I do not speak english"))
langid.set_languages(['de','fr','it'])
print(langid.classify("I do not speak english"))
print(langid.classify("Je ne parle pas français"))
"""" 
The probabilistic model implemented by langid.py involves the multiplication of a large number of probabilities. For 
computational reasons, the actual calculations are implemented in the log-probability space (a common numerical technique 
for dealing with vanishingly small probabilities). One side-effect of this is that it is not necessary to compute a full 
probability in order to determine the most probable language in a set of candidate languages. However, users sometimes 
find it helpful to have a "confidence" score for the probability prediction. Thus, langid.py implements a re-normalization 
that produces an output in the 0-1 range.
"""
from langid.langid import LanguageIdentifier, model
langid.set_languages(None)
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
print(identifier.classify("This is a test")) #After setting langid to None
from langdetect import detect, detect_langs
print(detect("War doesn't show who's right, just who's left."))
print(detect_langs("Otec matka syn."))





"""
Polygot supports about 165 languages. To install, type the following commands (in linux) - 
pip install pycld2 
pip install polyglot
And for Christ's sake, dont type - pip install pycld2 polyglot !!! (-_-) pycld2 is dependent on polyglot, so it wont work.
In case of windows, first unzip the following repo in your directory - 
https://github.com/aboSamoor/polyglot/archive/master.zip
Then, navigate to the  polyglot-master directory and run the following setup commands (cd ...bla-bla-bla...\polyglot-master)- 
python setup.py install
"""
from polyglot.detect import Detector
arabic_text = u"""
أفاد مصدر امني في قيادة عمليات صلاح الدين في العراق بأن " القوات الامنية تتوقف لليوم
الثالث على التوالي عن التقدم الى داخل مدينة تكريت بسبب
انتشار قناصي التنظيم الذي يطلق على نفسه اسم "الدولة الاسلامية" والعبوات الناسفة
والمنازل المفخخة والانتحاريين، فضلا عن ان القوات الامنية تنتظر وصول تعزيزات اضافية ".
"""
detector = Detector(arabic_text)
print(detector.language)

mixed_text = u"""
China (simplified Chinese: 中国; traditional Chinese: 中國),
officially the People's Republic of China (PRC), is a sovereign state located in East Asia.
"""
for language in Detector(mixed_text).languages:
  print(language)
detector = Detector("pizza")
print(detector)