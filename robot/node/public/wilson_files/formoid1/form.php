<?php

define('EMAIL_FOR_REPORTS', '');
define('RECAPTCHA_PRIVATE_KEY', '@privatekey@');
define('FINISH_URI', 'http://');
define('FINISH_ACTION', 'message');
define('FINISH_MESSAGE', 'XML Requested. ');
define('UPLOAD_ALLOWED_FILE_TYPES', 'doc, docx, xls, csv, txt, rtf, html, zip, jpg, jpeg, png, gif');

require_once str_replace('\\', '/', __DIR__) . '/handler.php';

?>

<?php if (frmd_message()): ?>
<link rel="stylesheet" href="<?=dirname($form_path)?>/formoid-flat-green.css" type="text/css" />
<span class="alert alert-success"><?=FINISH_MESSAGE;?></span>
<?php else: ?>
<!-- Start Formoid form-->
<link rel="stylesheet" href="<?=dirname($form_path)?>/formoid-flat-green.css" type="text/css" />
<script type="text/javascript" src="<?=dirname($form_path)?>/jquery.min.js"></script>
<form enctype="multipart/form-data" class="formoid-flat-green" style="background-color:#FFFFFF;font-size:14px;font-family:'Lato', sans-serif;color:#666666;max-width:480px;min-width:150px" method="post"><div class="title"><h2>Wilson Portal</h2></div>
	<div class="element-radio<?frmd_add_class("radio")?>"><label class="title">Template</label>		<div class="column column1"><label><input type="radio" name="radio" value="Event" /><span>Event</span></label><label><input type="radio" name="radio" value="Command" /><span>Command</span></label><label><input type="radio" name="radio" value="TestCase" /><span>TestCase</span></label><label><input type="radio" name="radio" value="NodeBus" /><span>NodeBus</span></label></div><span class="clearfix"></span>
</div>
	<div class="element-textarea<?frmd_add_class("textarea")?>"><label class="title">Write XML</label><textarea class="medium" name="textarea" cols="20" rows="5" ></textarea></div>
	<div class="element-file<?frmd_add_class("file")?>"><label class="title">XML File</label><label class="large" ><div class="button">Choose File</div><input type="file" class="file_input" name="file" /><div class="file_text">No file selected</div></label></div>
	<div class="element-input<?frmd_add_class("input1")?>"><label class="title">Search</label><input class="large" type="text" name="input1" /></div>
<div class="submit"><input type="submit" value="Request"/></div></form><script type="text/javascript" src="<?=dirname($form_path)?>/formoid-flat-green.js"></script>

<!-- Stop Formoid form-->
<?php endif; ?>

<?php frmd_end_form(); ?>