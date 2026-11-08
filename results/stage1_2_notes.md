# aşama 1 ve 2. aşamanın pilotu: notlar

8 kasım 2026. model qwen3.5-0.8b (752 milyon parametre, 24 katman, her katmanda 1024 sayılık bir vektör). veri olarak math-500'ün ilk 50 sorusunu kullandım. greedy decoding ile, düşünme modu kapalı, colab'da t4 gpu ile çalıştırdım.

## aşama 1: model ne sıklıkla takılıyor

önce 1536 token sınırıyla denedim, bitiremeyenler çok olunca sınırı 4096'ya çıkardım.

| | 1536 token | 4096 token |
|---|---|---|
| doğru | 14 | 20 (%40) |
| bitiremeyen | 23 | 19 (%38) |
| bunların içinde birebir tekrar | 1 | 4 |
| bunların içinde aynı fikirde dönüp durma | 22 | 15 |
| bitiremeyip doğru olan | 0 | 0 |

doğru sayılarını önce kaba bir eşleştirmeyle saymıştım. 4096'da 17 bulmuştum. mit'nin kullandığı math-verify ile yeniden puanlayınca 20 çıktı. 1536 sütunu hâlâ kaba sayım.

model aynı cümleyi kopyalayıp yapıştırmayı sadece 4 cevapta yaptı. kalanlarda bir şüpheye takılıp onun etrafında dönüp duruyor. 7. soruda "Is it possible the equation is 6x = -y AND -y = -4z? Yes." cümlesini küçük farklarla defalarca yazıyor. aslında daha önce bir cebir hatası yapmış ama hatayı aramak yerine soruyu tekrar tekrar okuyor.

## aşama 2 pilotu: modelin içinde bir sinyal var mı

her cevabın ilk 512 token'ını modele geri verip her katmanda ortalama aktivasyonu aldım. bir katmanda takılanların ortalamasından bitenlerin ortalamasını çıkarınca bir yön çıkıyor. bu yön takılmayı tahmin edebiliyor mu diye her cevabı sırayla dışarıda bırakıp kalanlarla tahmin ettirdim.

- hep "takılmaz" demek %62 doğru. yenmem gereken çizgi bu.
- katman 0'da %56, 4 ile 10 arası %70-74, **11. ve 17. katmanda %78**, son katmanlarda %72.
- yani sinyal en çok orta katlarda.

### dikkat etmem gerekenler

1. 50 örnek az. 25 katmandan en iyisini seçince sonuç olduğundan iyi görünebilir, çünkü bir tanesi şans eseri yüksek çıkmış olabilir.
2. takılan sorular aynı zamanda zor sorular olabilir. o zaman bulduğum yön takılmayla hiç ilgilenmiyor, sadece sorunun zor olduğunu gösteriyor olabilir.

### zorluk kontrolü

math-500'de her sorunun 1'den 5'e bir zorluk seviyesi var. takılmaya seviyeye göre baktım:

| seviye | soru | takılan |
|---|---|---|
| 1 | 4 | 0 (%0) |
| 2 | 13 | 4 (%31) |
| 3 | 12 | 3 (%25) |
| 4 | 8 | 5 (%62) |
| 5 | 13 | 7 (%54) |

sadece seviyeye bakarak tahmin edince en iyi %68 çıkıyor. iç aktivasyonla %78. zorluk takılmanın bir kısmını açıklıyor ama içeride zorluğun ötesinde bir şey de var gibi. yine de aradaki fark 50 soruda 5 soruya denk geliyor, buna güvenmek için erken.

## mit ile neden bu kadar farklıyım

mit math-500'de %7.9 raporladı, ben %40 buldum. ayarlarına baktım ([blog](https://rikiyat.github.io/blog/weight-space-programming/), [deney notları](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)):

| | mit | ben |
|---|---|---|
| düşünme modu | açık | kapalı |
| token sınırı | 32.768 | 4096 |
| seçim | temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5 | greedy |
| soru | 500 soru, her birine 4 deneme | 50 soru, birer deneme |
| puanlama | math-verify | math-verify |
| bitiremeyen | %93 | %38 |

asıl fark düşünme modu. açıkken model `<think>` içinde döngüye giriyor ve cevapların %93'ü 32 bin token'ın sonuna dayanıyor.

mit'nin düzenlemesi de benim yaptığımla aynı aileden. onlar da bir "loop direction" bulup ağırlıklardan çıkarmışlar: W' = W − 1.5·u(uᵀW), 8. ile 22. katmanlar arasında. en iyi sonucu 15. ve 19. katmanın mlp'sine dokunarak almışlar. benim pilotta sinyal en güçlü 11. ve 17. katmanda, yani aşağı yukarı aynı bölge.

## sırada ne var

1. aynısını 200 soruyla tekrarlamak. bunu yapıyorum.
2. yönü ağırlıklardan silip daha önce görmediği 100 yeni soruda takılma ve doğruluk değişiyor mu bakmak (`src/stage3_ablate.py`). zorluk sorusuna asıl cevabı bu verecek: yön sadece "zor soru" demekse silince bir şey değişmez.
