//***** ページの体裁調整 *****//
// bodyのマージン調整
$(function() {
  // フッター分引き上げる
  var footerHeight = $('footer').height();
  $('body').css('margin-bottom', footerHeight);
  // トップページ以外ではナビバー分引き下げる(重なっちゃうので)
  if (location.pathname != '/iis-lab/' || location.pathname != '/iis-lab/english.php') {
    var headerHeight = $('header').height();
    $('body').css('margin-top', headerHeight);
  }
});


// 英語トップページの名簿欄の幅調整(幅を広げないと崩れる)
$(function() {
  // CSSでも定義しているが，englishから戻った場合に書き直す必要あり
  if (location.pathname == '/iis-lab/') {
    $('.member-left').css('width', '25%');
    $('.member-left').css('margin-left', '25%');
    $('.member-right').css('width', '25%');
    $('.member-right').css('margin-right', '25%');
  }
  // englishの場合，幅変更
  else if (location.pathname == '/iis-lab/english.php') {
    $('.member-left').css('width', '35%');
    $('.member-left').css('margin-left', '15%');
    $('.member-right').css('width', '35%');
    $('.member-right').css('margin-right', '15%');
  }
});


// 自己満エリアの幅調整(英名のほうが長いので)
$(function() {
  // CSSでも定義しているが，englishから戻った場合に書き直す必要あり
  if (location.pathname == '/iis-lab/pages/member.php') {
    $('.editor').css('width', '15%');
  }
  // englishの場合，幅変更
  else if (location.pathname == '/iis-lab/pages/english/en-member.php') {
    $('.editor').css('width', '25%');
  }
});
//***** ここまで *****//



//***** トップページの下スクロールボタン *****//
// 下スクロールボタン
// 読み込み時は隠しておく
$(function() {
  $('.downs').hide();
  $('.down').hide();
});
// ムービー終了後に矢印ボタンを表示
$(function() {
  // クラス定義の関係で警告が出るためトップページでのみ実行
  if (location.pathname == '/iis-lab/' || location.pathname == '/iis-lab/english.php') {
    var video = document.getElementsByTagName('video')[0];
    video.addEventListener('ended', function() {
      $('.downs').fadeIn();
      $('.down').fadeIn();
    }, false);
  }
});


// スムーズスクロール
$(function() {
  $('a[href^="#"]').click(function() {
    var speed = 800,
    href= $(this).attr("href"),
    target = $(href == "#" || href == "" ? 'html' : href),
    position = target.offset().top;
    $("html, body").animate({scrollTop:position}, speed, "swing");
    return false;
  });
});
//***** ここまで *****//



//***** ナビバーの追従 *****//
$(function() {
  // トップページでは透過から黒へ変化をつける
  if (location.pathname == '/iis-lab/' || location.pathname == '/iis-lab/english.php') {
    var topHeight = $('.top').height(),
    headerHeight = $('header').height();
    $(window).scroll(function() {
      var scroll = $(this).scrollTop();
      if (scroll > topHeight) {
        $('header').addClass('is-fixed');
        $('.main').css('margin-top', headerHeight);
      }
      else {
        $('header').removeClass('is-fixed');
        $('.main').css('margin-top', '0');
      }
    });
  }
  // トップページ以外
  else {
    // 変化なし，固定色
    $('header').css('background-color', 'rgba(0, 0, 0, 0.9)');
  }
});
//***** ここまで *****//



//***** 効果など *****//
// 背景色の変化
$(function() {
  // トップページでは変化をつける
  if (location.pathname == '/iis-lab/' || location.pathname == '/iis-lab/english.php') {
    // 下にスクロールすると背景色を黒から白へ変更
    // 初期値を黒に設定
    $('body').css('background-color', 'black');
    $('.main').css('background-color', 'black');
    // 白に変更
    var topHeight = $('.top').height();
    $(window).scroll(function() {
      var scroll = $(this).scrollTop();
      if (scroll > topHeight*0.4) {
        $('body').animate({
          'background-color': '#f5f5f5'
        }, 'slow');
        $('.main').animate({
          'background-color': 'white'
        }, 'slow');
      }
    });
  }
  // トップページ以外
  else {
    // 変化なし，背景色固定
    $('body').css('background-color', '#f5f5f5');
    $('.main').css('background-color', 'white');
  }
});


// About Usのスクロールイン
$(function() {
  var topHeight = $('.top').height();
  $(window).scroll(function() {
    var scroll = $(this).scrollTop();
    // 詳細と画像のスクロールインに時間差をつける
    if (scroll > topHeight*0.6) {
      $('.detail').addClass('scrollin');
    }
    if (scroll > topHeight*0.7) {
      $('.pic').addClass('scrollin');
    }
  });
});
//***** ここまで *****//
