# küçük bir modelin içini açmak

## in english

in november 2026 mit showed that gpt-6 astra could raise qwen3.5-0.8b's math-500 score from 7.9% to 48.1% by editing a tiny part of its weights, mostly by stopping the model from looping. i'm trying to do the same by hand, on a free colab gpu, so i can see what is going on. so far, on 200 questions, looking only at the first 512 tokens, a direction in layer 15 predicts which answers will get stuck with 79.5% accuracy (64% by always guessing "won't loop", 68% from question difficulty alone). on 100 unseen questions, removing that direction raised the stuck answers from 34 to 72 and pushing the other way lowered them to 21, so the model really uses it to decide when to stop. fewer loops did not mean more correct answers though. a second direction built from repeated text, closer to what mit describes, neither broke nor fixed the model. my main suspect for the gap with mit is that i ran with thinking mode off. the first post is also [in english](https://furkankadiryilmaz.com/writing/the-model-knows-but-cant-decide/). the notes are in turkish, the code and the numbers should be readable either way.

## türkçe

kasım 2026'da mit'den bir [çalışma](https://rikiyat.github.io/blog/weight-space-programming/) çıktı. gpt-6 astra'ya qwen3.5-0.8b verilmiş ve astra ağırlıkların çok küçük bir kısmına dokunmuş ve modelin math-500 skoru %7.9'dan %48.1'e çıkmış. kazancın büyük kısmı modelin takılıp aynı şeyi tekrar etmeyi bırakmasından geliyormuş.

ben aynı şeyi astra'yı kullanmadan, elle ve küçük ölçekte yapmaya çalışıyorum. amacım ne olduğunu kendim görüp anlamak. sonra da aynı şey türkçe sorularda ve türkçe modellerde oluyor mu, ona bakacağım.

ilerledikçe her aşamayı [blogumda](https://furkankadiryilmaz.com/tr/yazilar/) yazıyorum. bu depo o yazıların arkasındaki kod ve notlar.

### şu ana kadar

- math-500'ün ilk 50 sorusunda 4096 token sınırıyla model 20 soruyu doğru çözdü. 19 cevap hiç bitmedi, bitmeyenlerin hiçbiri doğru değildi.
- 200 soruda cevabın ilk 512 token'ına bakarak modelin takılıp takılmayacağını 15. katmanda %79.5 doğrulukla tahmin edebildim. hep "takılmaz" demek %64, sadece sorunun zorluğuna bakmak %68 veriyor.
- bu yön modelin bitirme kararında gerçekten kullanılıyor. modelin hiç görmediği 100 soruda yönü silince takılan cevap 34'ten 72'ye çıktı, ters tarafa itince 21'e indi.
- ama takılmayı azaltmak doğru sayısını artırmadı. model bitirmeye zorlanınca emin olmadığı cevabı da yazıp geçiyor.
- mit'nin yöntemini cevaptaki tekrarlardan bulduğum ayrı bir yönle de denedim. model ne bozuldu ne düzeldi. mit'nin büyük kazancını bu ayarlarda göremedim, en büyük şüphem düşünme modunun kapalı olması.

ayrıntılar [aşama 1 ve pilot notlarında](results/stage1_2_notes.md) ve [aşama 2 ve 3 notlarında](results/stage2_3_notes.md).

ilk yazı: [model biliyor ama karar veremiyor](https://furkankadiryilmaz.com/tr/yazilar/model-bilmedigi-icin-degil-kararsiz-kaldigi-icin-kaybediyor/)

### aşamalar

1. modelin ne sıklıkla takıldığını saymak. bitti.
2. takılmanın modelin içinde bir yönü var mı, bulmak. bitti.
3. o yönü ağırlıklardan silip takılma ve doğruluk değişiyor mu bakmak. bitti, sırada aynısını düşünme modu açıkken denemek var.
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
- `data/`: modelin ham cevapları. `stage1_raw_4096.json` ilk 50 sorunun bütün cevaplarını tutuyor, her cevabın token'ları, metni ve bitip bitmediği içinde. `stage2_200_raw.json` 200 soruyu, `stage3_*.json` 3. aşamanın her denemesini, `u_layer15.npy` 15. katmandaki yönü tutuyor.

### kaynaklar

- mit'nin [blog yazısı](https://rikiyat.github.io/blog/weight-space-programming/) ve [deney notları](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)
- arditi ve arkadaşları, [refusal in language models is mediated by a single direction](https://arxiv.org/abs/2406.11717). yön bulup ağırlıklardan silme yöntemi buradan geliyor.
- barış bayburt'un [kumru üzerine yazısı](https://blog.bayburt.lu/editing-kumru-weights). astra ile kumru-2b'nin ağırlıklarını düzenlemiş.

### lisans

kod mit lisanslı, bkz. `LICENSE`.
