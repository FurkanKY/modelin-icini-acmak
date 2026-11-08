# küçük bir modelin içini açmak

## in english

in november 2026 mit showed that gpt-6 astra could raise qwen3.5-0.8b's math-500 score from 7.9% to 48.1% by editing a tiny part of its weights, mostly by stopping the model from looping. i'm trying to do the same by hand, on a free colab gpu, so i can see what is going on. so far, on the first 50 questions, 19 answers never finished and none of those was correct. looking only at the first 512 tokens, a direction in the middle layers predicts which answers will get stuck with 78% accuracy (62% by always guessing "won't loop", 68% from question difficulty alone). next i'm checking this on 200 questions, removing that direction from the weights, and then asking the same question for turkish. the first post is also [in english](https://furkankadiryilmaz.com/writing/the-model-knows-but-cant-decide/). the notes are in turkish, the code and the numbers should be readable either way.

## türkçe

kasım 2026'da mit'den bir [çalışma](https://rikiyat.github.io/blog/weight-space-programming/) çıktı. gpt-6 astra'ya qwen3.5-0.8b verilmiş ve astra ağırlıkların çok küçük bir kısmına dokunmuş ve modelin math-500 skoru %7.9'dan %48.1'e çıkmış. kazancın büyük kısmı modelin takılıp aynı şeyi tekrar etmeyi bırakmasından geliyormuş.

ben aynı şeyi astra'yı kullanmadan, elle ve küçük ölçekte yapmaya çalışıyorum. amacım ne olduğunu kendim görüp anlamak. sonra da aynı şey türkçe sorularda ve türkçe modellerde oluyor mu, ona bakacağım.

ilerledikçe her aşamayı [blogumda](https://furkankadiryilmaz.com/tr/yazilar/) yazıyorum. bu depo o yazıların arkasındaki kod ve notlar.

### şu ana kadar

- math-500'ün ilk 50 sorusunda 4096 token sınırıyla model 20 soruyu doğru çözdü. 19 cevap hiç bitmedi, bitmeyenlerin hiçbiri doğru değildi.
- cevabın ilk 512 token'ına bakarak modelin takılıp takılmayacağını orta katlarda %78 doğrulukla tahmin edebildim. hep "takılmaz" demek %62, sadece sorunun zorluğuna bakmak %68 veriyor.
- bu 50 soruluk bir pilot. şu an aynısını 200 soruyla tekrarlıyorum.

ilk yazı: [model biliyor ama karar veremiyor](https://furkankadiryilmaz.com/tr/yazilar/model-bilmedigi-icin-degil-kararsiz-kaldigi-icin-kaybediyor/)

### aşamalar

1. modelin ne sıklıkla takıldığını saymak. bitti.
2. takılmanın modelin içinde bir yönü var mı, bulmak. pilot bitti, 200 soru sürüyor.
3. o yönü ağırlıklardan silip takılma ve doğruluk değişiyor mu bakmak. kod hazır.
4. aynı yön türkçe sorularda da var mı.
5. aynısını kumru-2b ya da karga-2b-thinking gibi türkçe modellerde denemek.

### nasıl çalıştırılır

her şey colab'da, t4 gpu ile çalıştı. kodlar aynı colab oturumunda sırayla hücre hücre çalıştırıldı, yani tek başına çalışan scriptler değil. biri diğerinin değişkenlerini kullanıyor.

```
pip install -U transformers accelerate datasets math-verify
```

- `src/stage1_generate.py`: modeli yükler, soruları sorar, cevapları kaydeder
- `src/stage2_pilot.py`: cevapların iç aktivasyonlarını toplar, katman katman yön bulur ve test eder
- `src/stage3_ablate.py`: yönü ağırlıklardan siler, yeni sorularda tekrar ölçer

ben greedy decoding kullandım, düşünme modunu kapattım ve sınırı 4096 token'da tuttum. mit ise düşünme modu açık, 32.768 token ve sampling ile çalıştı. o yüzden sayılarım onlarınkiyle birebir karşılaştırılamaz, farkı [notlarda](results/stage1_2_notes.md) anlattım.

### dosyalar

- `src/`: kodlar
- `results/`: her aşamanın sayıları ve notları
- `blog/`: yazıların taslakları ve grafikler
- `data/`: modelin ham cevapları. `stage1_raw_4096.json` ilk 50 sorunun bütün cevaplarını tutuyor, her cevabın token'ları, metni ve bitip bitmediği içinde.

### kaynaklar

- mit'nin [blog yazısı](https://rikiyat.github.io/blog/weight-space-programming/) ve [deney notları](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)
- arditi ve arkadaşları, [refusal in language models is mediated by a single direction](https://arxiv.org/abs/2406.11717). yön bulup ağırlıklardan silme yöntemi buradan geliyor.
- barış bayburt'un [kumru üzerine yazısı](https://blog.bayburt.lu/editing-kumru-weights). astra ile kumru-2b'nin ağırlıklarını düzenlemiş.

### lisans

kod mit lisanslı, bkz. `LICENSE`.
