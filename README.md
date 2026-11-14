# küçük bir modelin içini açmak

## in english

in november 2026 mit showed that gpt-6 astra could raise qwen3.5-0.8b's math-500 score from 7.9% to 48.1% by editing a tiny part of its weights, mostly by stopping the model from looping. i'm trying to understand what is going on by hand, on a free colab gpu, and then asking the same questions in turkish.

what i found so far:

- a direction in layer 15 predicts which answers will get stuck with 79.5% accuracy from the first 512 tokens (64% by always guessing "won't loop", 68% from question difficulty alone). removing it raised stuck answers from 34 to 72 on 100 unseen questions, pushing the other way lowered them to 21. fewer loops did not mean more correct answers though.
- with thinking mode on, 95 of 100 answers fell into a loop that copies the same lines. cutting the loop from the outside took correct answers from 10 to 35. i couldn't find a direction inside the model that breaks it.
- translating words from french and german into turkish, qwen keeps the concept close to english and just as close to chinese, the two languages it has seen the most. deleting either one takes about a fifth of the right answer's probability in 0.8b, 2b and 4b, deleting german or french doesn't. kumru-2b, trained from scratch for turkish, doesn't need english at all.

every step is a post on my blog, in turkish and english:

1. [the model knows but can't decide](https://furkankadiryilmaz.com/writing/the-model-knows-but-cant-decide/)
2. [the model has a finish button](https://furkankadiryilmaz.com/writing/the-model-has-a-finish-button/)
3. [with thinking mode on, the model copies itself](https://furkankadiryilmaz.com/writing/thinking-mode-makes-the-model-copy-itself/)
4. [does a turkish-speaking model think in english?](https://furkankadiryilmaz.com/writing/does-a-turkish-speaking-model-think-in-english/)

the notes are in turkish, the code and the numbers should be readable either way.

## türkçe

kasım 2026'da mit'den bir [çalışma](https://rikiyat.github.io/blog/weight-space-programming/) çıktı. gpt-6 astra'ya qwen3.5-0.8b verilmiş, astra ağırlıkların çok küçük bir kısmına dokunmuş ve modelin math-500 skoru %7.9'dan %48.1'e çıkmış. kazancın büyük kısmı modelin takılıp aynı şeyi tekrar etmeyi bırakmasından geliyormuş.

ben aynı şeyi astra'yı kullanmadan, elle ve küçük ölçekte anlamaya çalışıyorum. sonra da aynı soruları türkçe için ve türkçe modellerde soruyorum.

ilerledikçe her aşamayı [blogumda](https://furkankadiryilmaz.com/tr/yazilar/) yazıyorum. bu depo o yazıların arkasındaki kod ve notlar.

### yazılar

1. [model biliyor ama karar veremiyor](https://furkankadiryilmaz.com/tr/yazilar/model-bilmedigi-icin-degil-kararsiz-kaldigi-icin-kaybediyor/)
2. [modelin bir bitirme düğmesi var](https://furkankadiryilmaz.com/tr/yazilar/modelin-bir-bitirme-dugmesi-var/)
3. [düşünme modunu açınca model kendini kopyalıyor](https://furkankadiryilmaz.com/tr/yazilar/dusunme-modunu-acinca-model-kendini-kopyaliyor/)
4. [türkçe konuşan model kafasında ingilizce mi düşünüyor?](https://furkankadiryilmaz.com/tr/yazilar/turkce-konusan-model-kafasinda-ingilizce-mi-dusunuyor/)

### şu ana kadar

- math-500'ün ilk 50 sorusunda 4096 token sınırıyla model 20 soruyu doğru çözdü. 19 cevap hiç bitmedi, bitmeyenlerin hiçbiri doğru değildi.
- 200 soruda cevabın ilk 512 token'ına bakarak modelin takılıp takılmayacağını 15. katmanda %79.5 doğrulukla tahmin edebildim. hep "takılmaz" demek %64, sadece sorunun zorluğuna bakmak %68 veriyor.
- bu yön modelin bitirme kararında gerçekten kullanılıyor. modelin hiç görmediği 100 soruda yönü silince takılan cevap 34'ten 72'ye çıktı, ters tarafa itince 21'e indi. takılmayı azaltmak doğru sayısını artırmadı, model bitirmeye zorlanınca emin olmadığı cevabı da yazıp geçiyor.
- düşünme modu açıkken 100 cevabın 95'i aynı satırları kopyalayan bir döngüye girdi. döngüyü dışarıdan kesince doğru sayısı 10'dan 35'e çıktı. modelin içinde onu kıran bir yön bulamadım.
- fransızcadan ve almancadan türkçeye kelime çevirirken qwen kavramı ingilizceye yakın bir yerde tutuyor, çinceye de bir o kadar. ikisinden birini silmek 0.8b, 2b ve 4b'de doğru cevabın olasılığının aşağı yukarı beşte birini götürüyor, almancayı ya da fransızcayı silmek götürmüyor. sıfırdan türkçe için eğitilmiş kumru-2b çeviri yaparken ingilizceye hiç ihtiyaç duymuyor.

ayrıntılar notlarda: [aşama 1 ve pilot](results/stage1_2_notes.md), [aşama 2 ve 3](results/stage2_3_notes.md), [aşama 4](results/stage4_notes.md).

### aşamalar

1. modelin ne sıklıkla takıldığını saymak. bitti.
2. takılmanın modelin içinde bir yönü var mı, bulmak. bitti.
3. o yönü ağırlıklardan silip takılma ve doğruluk değişiyor mu bakmak, düşünme modu kapalı ve açıkken. bitti.
4. model türkçe çevirirken içeride ingilizceden geçiyor mu, qwen3.5 ve kumru-2b'de. bitti.
5. türkçe sorularda da aynı takılma oluyor mu.

### nasıl çalıştırılır

her şey colab'da, t4 gpu ile çalıştı. 1-3. aşamaların kodları aynı colab oturumunda sırayla hücre hücre çalıştırıldı, yani tek başına çalışan scriptler değil. biri diğerinin değişkenlerini kullanıyor.

```
pip install -U transformers accelerate datasets math-verify
```

- `src/stage1_generate.py`: modeli yükler, soruları sorar, cevapları kaydeder
- `src/stage2_pilot.py`: cevapların iç aktivasyonlarını toplar, katman katman yön bulur ve test eder
- `src/stage3_*.py`: yönü ağırlıklardan siler, yeni sorularda tekrar ölçer, düşünme modunu açar
- `src/stage4_*.py`: çeviri deneyi. `stage4_latent.py` logit lens, `stage4_tunedlens.py` tuned lens, `stage4_bridge.py` kelimeye özel silme, `stage4_source.py` kaynak kelime kontrolü, `stage4_boyut.py` üç boyutun karşılaştırması, `stage4_zh.py` çince kontrolü, `stage4_suffix.py` ekli kelimeler
- `src/istatistik.py`: bootstrap aralıkları ve mcnemar testi

1-3. aşamalarda greedy decoding kullandım ve sınırı 4096 token'da tuttum. mit ise düşünme modu açık, 32.768 token ve sampling ile çalıştı. o yüzden sayılarım onlarınkiyle birebir karşılaştırılamaz, farkı [notlarda](results/stage1_2_notes.md) anlattım.

### dosyalar

- `src/`: kodlar
- `results/`: her aşamanın sayıları ve notları
- `blog/`: yazıların taslakları ve grafikler
- `data/`: modelin ham cevapları ve ölçümler. `stage1_raw_4096.json` ilk 50 sorunun bütün cevaplarını tutuyor, her cevabın token'ları, metni ve bitip bitmediği içinde. `stage2_200_raw.json` 200 soruyu, `stage3_*.json` 3. aşamanın her denemesini, `stage4_*.json` çeviri deneylerini, `u_layer15.npy` 15. katmandaki yönü tutuyor.

### kaynaklar

- mit'nin [blog yazısı](https://rikiyat.github.io/blog/weight-space-programming/) ve [deney notları](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)
- arditi 2024, [refusal in language models is mediated by a single direction](https://arxiv.org/abs/2406.11717). yön bulup ağırlıklardan silme yöntemi buradan geliyor.
- wendler 2024, [do llamas work in english?](https://arxiv.org/abs/2402.10588). 4. aşamadaki çeviri kurgusu buradan.
- belrose 2023, [eliciting latent predictions with the tuned lens](https://arxiv.org/abs/2303.08112)
- barış bayburtlu'nun [kumru üzerine yazısı](https://blog.bayburt.lu/editing-kumru-weights). astra ile kumru-2b'nin ağırlıklarını düzenlemiş.
